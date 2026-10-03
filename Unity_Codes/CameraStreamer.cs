using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System;
using System.Threading;
using System.Collections.Concurrent;

public class PipelineCameraUDP : MonoBehaviour
{
    public int port = 5005;
    public int width = 640;
    public int height = 480;
    public int targetFPS = 30;
    public string serverIP = "127.0.0.1";
    public int jpegQuality = 70;
    public int maxPacketSize = 65507; // Maximum UDP packet size
    
    private Camera cam;
    private UdpClient udpClient;
    private IPEndPoint serverEndPoint;
    private Thread sendThread;
    private bool isRunning = true;
    private float frameInterval;
    private float nextFrameTime;
    
    // Single queue for JPEG data
    private ConcurrentQueue<byte[]> sendQueue = new ConcurrentQueue<byte[]>();
    
    // For performance monitoring
    private int framesSent = 0;
    private float lastFpsLogTime;
    private int sequenceNumber = 0;

    void Start()
    {
        cam = GetComponent<Camera>();
        Application.runInBackground = true;
        frameInterval = 1f / targetFPS;
        nextFrameTime = Time.time;
        
        try
        {
            // Convert server IP to IPAddress
            IPAddress ipAddress = IPAddress.Parse(serverIP);
            serverEndPoint = new IPEndPoint(ipAddress, port);
            
            // Create UDP client
            udpClient = new UdpClient();
            
            // Optional: Set buffer sizes for better performance
            udpClient.Client.SendBufferSize = 1024 * 1024; // 1MB send buffer
            udpClient.Client.ReceiveBufferSize = 1024 * 1024; // 1MB receive buffer
            
            Debug.Log($"📷 PipelineCameraUDP ready to send to {serverIP}:{port}");
            
            // Start send thread
            sendThread = new Thread(SendLoop);
            sendThread.Start();
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to initialize UDP: {e.Message}");
        }
    }

    void Update()
    {
        if (udpClient == null)
            return;

        // --- FRAME RATE LIMITING ---
        if (Time.time < nextFrameTime)
            return;
        
        nextFrameTime = Time.time + frameInterval;

        // --- CAPTURE AND ENCODE ---
        RenderTexture rt = RenderTexture.GetTemporary(width, height, 24);
        cam.targetTexture = rt;
        cam.Render();

        Texture2D tex = new Texture2D(width, height, TextureFormat.RGB24, false);
        RenderTexture.active = rt;
        tex.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        tex.Apply();

        // Encode to JPEG
        byte[] jpegData = tex.EncodeToJPG(jpegQuality);
        
        // Cleanup
        cam.targetTexture = null;
        RenderTexture.active = null;
        RenderTexture.ReleaseTemporary(rt);
        Destroy(tex);

        // Only enqueue if queue isn't too backed up
        if (sendQueue.Count < 5)
        {
            sendQueue.Enqueue(jpegData);
        }
        else
        {
            Debug.LogWarning($"Queue full ({sendQueue.Count}), skipping frame");
        }

        // Performance logging
        framesSent++;
        if (Time.time - lastFpsLogTime > 2f)
        {
            Debug.Log($"📊 FPS: {framesSent / (Time.time - lastFpsLogTime):F1}, Queue: {sendQueue.Count}");
            framesSent = 0;
            lastFpsLogTime = Time.time;
        }
    }

    void SendLoop()
    {
        while (isRunning)
        {
            if (sendQueue.TryDequeue(out byte[] jpegData))
            {
                try
                {
                    // Add sequence number for packet ordering/reassembly
                    int seqNum = System.Threading.Interlocked.Increment(ref sequenceNumber);
                    
                    // Prepare packet: [sequence number (4 bytes)] + [JPEG data]
                    byte[] packetData = new byte[jpegData.Length + 4];
                    byte[] seqBytes = BitConverter.GetBytes(seqNum);
                    Array.Copy(seqBytes, 0, packetData, 0, 4);
                    Array.Copy(jpegData, 0, packetData, 4, jpegData.Length);
                    
                    // If packet is too large for UDP, split it
                    if (packetData.Length > maxPacketSize)
                    {
                        SendLargePacket(packetData);
                    }
                    else
                    {
                        // Send single packet
                        udpClient.Send(packetData, packetData.Length, serverEndPoint);
                    }
                }
                catch (Exception e)
                {
                    Debug.LogError($"Send error: {e.Message}");
                }
            }
            else
            {
                // No data, sleep briefly
                Thread.Sleep(1);
            }
        }
    }

    void SendLargePacket(byte[] data)
    {
        // Split large packets into chunks
        int totalChunks = (data.Length + maxPacketSize - 1) / maxPacketSize;
        int chunkIndex = 0;
        int offset = 0;
        
        while (offset < data.Length)
        {
            int chunkSize = Math.Min(maxPacketSize, data.Length - offset);
            byte[] chunkData = new byte[chunkSize + 8]; // 8 bytes header: [sequence][chunk index][total chunks]
            
            // Header: [sequence number (4 bytes)] [chunk index (2 bytes)] [total chunks (2 bytes)]
            byte[] seqBytes = BitConverter.GetBytes(sequenceNumber);
            byte[] chunkIndexBytes = BitConverter.GetBytes((ushort)chunkIndex);
            byte[] totalChunksBytes = BitConverter.GetBytes((ushort)totalChunks);
            
            Array.Copy(seqBytes, 0, chunkData, 0, 4);
            Array.Copy(chunkIndexBytes, 0, chunkData, 4, 2);
            Array.Copy(totalChunksBytes, 0, chunkData, 6, 2);
            Array.Copy(data, offset, chunkData, 8, chunkSize);
            
            try
            {
                udpClient.Send(chunkData, chunkData.Length, serverEndPoint);
            }
            catch (Exception e)
            {
                Debug.LogError($"Failed to send chunk {chunkIndex}: {e.Message}");
            }
            
            offset += chunkSize;
            chunkIndex++;
            
            // Small delay between chunks to prevent flooding
            if (chunkIndex < totalChunks)
            {
                Thread.Sleep(1);
            }
        }
    }

    void OnDestroy()
    {
        isRunning = false;
        
        // Clear queues
        while (sendQueue.TryDequeue(out _)) { }
        
        if (udpClient != null)
        {
            try 
            { 
                udpClient.Close(); 
                udpClient.Dispose();
            } 
            catch { }
        }
        
        if (sendThread != null && sendThread.IsAlive)
            sendThread.Join(1000);
    }
}