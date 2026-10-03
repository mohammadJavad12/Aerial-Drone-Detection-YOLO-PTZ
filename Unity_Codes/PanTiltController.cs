using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System;
using System.Globalization;

public class PanTiltController : MonoBehaviour
{
    [Header("Optional Two Pivot Setup")]
    public Transform panPivot;
    public Transform tiltPivot;

    [Header("Limits")]
    public float minPan = -180f;
    public float maxPan = 180f;
    public float minTilt = -90f;
    public float maxTilt = 90f;

    [Header("Direction")]
    public bool invertPan = false;
    public bool invertTilt = false;

    public float sensitivity = 1f;

    [Header("Python")]
    public string pythonIP = "127.0.0.1";
    public int pythonStatePort = 8889;

    private UdpClient udpServer;
    private IPEndPoint clientEndpoint;

    private UdpClient stateSender;
    private IPEndPoint pythonEndpoint;

    private float targetPan = 0f;
    private float targetTilt = 0f;

    private readonly object angleLock = new object();

    void Start()
    {
        udpServer = new UdpClient(8888);
        clientEndpoint = new IPEndPoint(IPAddress.Any, 0);
        udpServer.BeginReceive(OnReceive, null);

        stateSender = new UdpClient();
        pythonEndpoint = new IPEndPoint(
            IPAddress.Parse(pythonIP),
            pythonStatePort
        );

        Debug.Log("PTZ Started");
    }

    void OnReceive(IAsyncResult result)
    {
        try
        {
            byte[] data = udpServer.EndReceive(result, ref clientEndpoint);

            string msg = Encoding.ASCII.GetString(data).Trim();

            if (msg.StartsWith("PAN_TILT"))
            {
                string[] parts = msg.Substring(8).Trim().Split(' ');

                if (parts.Length >= 2)
                {
                    float pan = float.Parse(parts[0], CultureInfo.InvariantCulture);
                    float tilt = float.Parse(parts[1], CultureInfo.InvariantCulture);

                    if (invertPan)
                        pan = -pan;

                    if (invertTilt)
                        tilt = -tilt;

                    lock (angleLock)
                    {
                        targetPan += pan * sensitivity;
                        targetTilt += tilt * sensitivity;

                        targetPan = Mathf.Clamp(targetPan, minPan, maxPan);
                        targetTilt = Mathf.Clamp(targetTilt, minTilt, maxTilt);
                    }
                }
            }

            udpServer.BeginReceive(OnReceive, null);
        }
        catch (Exception e)
        {
            Debug.LogError(e);

            try
            {
                udpServer.BeginReceive(OnReceive, null);
            }
            catch { }
        }
    }

    void Update()
    {
        float pan;
        float tilt;

        lock (angleLock)
        {
            pan = targetPan;
            tilt = targetTilt;
        }

        ApplyRotation(pan, tilt);

        SendCurrentState();
    }

    void ApplyRotation(float pan, float tilt)
    {
        if (panPivot != null && tiltPivot != null)
        {
            panPivot.localRotation = Quaternion.Euler(0f, pan, 0f);
            tiltPivot.localRotation = Quaternion.Euler(tilt, 0f, 0f);
        }
        else
        {
            transform.localRotation = Quaternion.Euler(tilt, pan, 0f);
        }
    }

    float GetCurrentPan()
    {
        Transform t = panPivot != null ? panPivot : transform;

        float y = t.localEulerAngles.y;

        if (y > 180f)
            y -= 360f;

        return y;
    }

    float GetCurrentTilt()
    {
        Transform t = tiltPivot != null ? tiltPivot : transform;

        float x = t.localEulerAngles.x;

        if (x > 180f)
            x -= 360f;

        return x;
    }

    void SendCurrentState()
    {
        try
        {
            string msg =
                $"STATE {GetCurrentPan().ToString("F4", CultureInfo.InvariantCulture)} " +
                $"{GetCurrentTilt().ToString("F4", CultureInfo.InvariantCulture)}";

            byte[] bytes = Encoding.ASCII.GetBytes(msg);

            stateSender.Send(bytes, bytes.Length, pythonEndpoint);
        }
        catch
        {
        }
    }

    void OnDestroy()
    {
        udpServer?.Close();
        stateSender?.Close();
    }
}