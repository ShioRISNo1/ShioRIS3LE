#ifndef WEB_SERVER_H
#define WEB_SERVER_H

#include <QObject>
#include <QTcpServer>
#include <QTcpSocket>
#include <QSslSocket>
#include <QSslCertificate>
#include <QSslKey>
#include <QMap>
#include <QByteArray>
#include <QMutex>
#include <functional>

class DicomViewer;
class WebServer;

/**
 * @brief SSL-enabled TCP server that overrides incomingConnection
 */
class SslServer : public QTcpServer {
    Q_OBJECT

public:
    explicit SslServer(WebServer* webServer, QObject* parent = nullptr);

    void setSSLConfiguration(const QSslCertificate& certificate, const QSslKey& privateKey);

protected:
    void incomingConnection(qintptr socketDescriptor) override;

private:
    WebServer* m_webServer;
    QSslCertificate m_sslCertificate;
    QSslKey m_sslPrivateKey;
};

/**
 * @brief Simple HTTP server for serving web interface and API
 *
 * This server provides:
 * - Static file serving (HTML, JS, CSS)
 * - REST API endpoints for DICOM data access
 * - Support for Vision Pro WebXR client
 */
class WebServer : public QObject {
    Q_OBJECT

public:
    explicit WebServer(DicomViewer* viewer, QObject* parent = nullptr);
    ~WebServer();

    /**
     * @brief Start the HTTP/HTTPS server on specified port
     * @param port Port number (default: 8443 for HTTPS, 8080 for HTTP)
     * @param useSSL Enable HTTPS with SSL/TLS encryption
     * @return true if server started successfully
     */
    bool start(quint16 port = 8443, bool useSSL = true);

    /**
     * @brief Stop the HTTP server
     */
    void stop();

    /**
     * @brief Check if server is running
     */
    bool isRunning() const;

    /**
     * @brief Get the server port
     */
    quint16 port() const;

    /**
     * @brief Check if SSL/TLS is enabled
     */
    bool isSSLEnabled() const;

    /**
     * @brief Set Window/Level for CT images (synchronizes with MainWindow)
     * @param window Window width
     * @param level Window center/level
     */
    void setWindowLevel(double window, double level);

    /**
     * @brief Get current Window/Level settings
     * @param window Output: current window width
     * @param level Output: current window center/level
     */
    void getWindowLevel(double& window, double& level) const;

    /**
     * @brief Handle new SSL socket (called by SslServer)
     */
    void handleNewSslSocket(QSslSocket* socket);

signals:
    void serverStarted(quint16 port);
    void serverStopped();
    void clientConnected(const QString& address);
    void requestReceived(const QString& method, const QString& path);

private slots:
    void onNewConnection();
    void onReadyRead();
    void onDisconnected();

private:
    struct HttpRequest {
        QString method;      // GET, POST, etc.
        QString path;        // /api/patients
        QString version;     // HTTP/1.1
        QMap<QString, QString> headers;
        QByteArray body;
    };

    struct HttpResponse {
        int statusCode = 200;
        QString statusMessage = "OK";
        QMap<QString, QString> headers;
        QByteArray body;
    };

    /**
     * @brief Parse HTTP request from raw data
     */
    HttpRequest parseRequest(const QByteArray& data);

    /**
     * @brief Handle incoming HTTP request
     */
    HttpResponse handleRequest(const HttpRequest& request);

    /**
     * @brief Send HTTP response to client
     */
    void sendResponse(QTcpSocket* socket, const HttpResponse& response);

    /**
     * @brief Serve static files (HTML, JS, CSS)
     */
    HttpResponse serveStaticFile(const QString& path);

    /**
     * @brief Handle API requests
     */
    HttpResponse handleApiRequest(const HttpRequest& request);

    /**
     * @brief Get MIME type from file extension
     */
    QString getMimeType(const QString& path) const;

    /**
     * @brief Load SSL certificate and private key
     * @return true if certificates loaded successfully
     */
    bool loadSSLCertificates();

    // API endpoint handlers
    HttpResponse apiGetPatients();
    HttpResponse apiGetVolume(const QString& patientId);
    HttpResponse apiGetStructures(const QString& patientId, const QMap<QString, QString>& queryParams = QMap<QString, QString>());
    HttpResponse apiGetDose(const QString& patientId);
    HttpResponse apiGetSlice(const QString& patientId, const QString& orientation, int sliceIndex, const QMap<QString, QString>& queryParams = QMap<QString, QString>());
    HttpResponse apiGetWindowLevel();

    QTcpServer* m_server;
    DicomViewer* m_viewer;
    quint16 m_port;
    bool m_useSSL;
    QSslCertificate m_sslCertificate;
    QSslKey m_sslPrivateKey;
    QMap<QTcpSocket*, QByteArray> m_socketBuffers;
    QMutex m_socketBuffersMutex;  // Protects m_socketBuffers from concurrent access

    // Window/Level settings (synchronized with MainWindow)
    double m_currentWindow;
    double m_currentLevel;
    QMutex m_windowLevelMutex;  // Protects window/level from concurrent access
};

#endif // WEB_SERVER_H
