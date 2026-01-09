#include "web/web_server.h"
#include "visualization/dicom_viewer.h"
#include "dicom/dicom_volume.h"
#include "dicom/rtstruct.h"
#include <QFile>
#include <QFileInfo>
#include <QDir>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QDateTime>
#include <QDate>
#include <QBuffer>
#include <QCoreApplication>
#include <QDebug>
#include <QSslSocket>
#include <QSslCertificate>
#include <QSslKey>
#include <QSslConfiguration>
#include <QSslCipher>

// ============================================================================
// SslServer Implementation
// ============================================================================

SslServer::SslServer(WebServer* webServer, QObject* parent)
    : QTcpServer(parent)
    , m_webServer(webServer)
{
}

void SslServer::setSSLConfiguration(const QSslCertificate& certificate, const QSslKey& privateKey) {
    m_sslCertificate = certificate;
    m_sslPrivateKey = privateKey;
}

void SslServer::incomingConnection(qintptr socketDescriptor) {
    qDebug() << "=== SslServer::incomingConnection called ===";
    qDebug() << "Socket descriptor:" << socketDescriptor;

    // Create SSL socket
    QSslSocket* sslSocket = new QSslSocket(this);

    if (!sslSocket->setSocketDescriptor(socketDescriptor)) {
        qCritical() << "Failed to set socket descriptor:" << sslSocket->errorString();
        delete sslSocket;
        return;
    }

    qDebug() << "Socket descriptor set successfully";
    qDebug() << "Client address:" << sslSocket->peerAddress().toString();

    // Configure SSL
    qDebug() << "Setting certificate and private key...";
    sslSocket->setLocalCertificate(m_sslCertificate);
    sslSocket->setPrivateKey(m_sslPrivateKey);

    // Configure SSL protocol
    QSslConfiguration sslConfig = sslSocket->sslConfiguration();
    sslConfig.setPeerVerifyMode(QSslSocket::VerifyNone);
    sslConfig.setProtocol(QSsl::TlsV1_2OrLater);
    sslConfig.setCiphers(QSslConfiguration::supportedCiphers());
    sslSocket->setSslConfiguration(sslConfig);

    qDebug() << "Socket has local certificate:" << !sslSocket->localCertificate().isNull();
    qDebug() << "Socket has private key:" << !sslSocket->privateKey().isNull();

    // Connect signals before starting encryption
    connect(sslSocket, &QSslSocket::encrypted, this, [this, sslSocket]() {
        qInfo() << "=== SSL Handshake Completed Successfully ===";
        qInfo() << "Encrypted connection from:" << sslSocket->peerAddress().toString();

        // Hand off the encrypted socket to WebServer
        m_webServer->handleNewSslSocket(sslSocket);
    });

    connect(sslSocket, QOverload<const QList<QSslError>&>::of(&QSslSocket::sslErrors),
            this, [sslSocket](const QList<QSslError>& errors) {
        qWarning() << "=== SSL Errors ===";
        for (const QSslError& error : errors) {
            qWarning() << "  SSL Error:" << error.errorString();
        }
    });

    connect(sslSocket, QOverload<QAbstractSocket::SocketError>::of(&QSslSocket::errorOccurred),
            this, [sslSocket](QAbstractSocket::SocketError error) {
        qCritical() << "=== SSL Socket Error ===";
        qCritical() << "Error code:" << error;
        qCritical() << "Error string:" << sslSocket->errorString();
    });

    connect(sslSocket, &QSslSocket::stateChanged, this, [](QAbstractSocket::SocketState state) {
        qDebug() << "SSL Socket state changed to:" << state;
    });

    // Start SSL handshake
    qDebug() << "Starting SSL server encryption...";
    qDebug() << "Socket state before encryption:" << sslSocket->state();
    sslSocket->startServerEncryption();
    qDebug() << "startServerEncryption() called";
    qDebug() << "Socket state after call:" << sslSocket->state();
}

// ============================================================================
// WebServer Implementation
// ============================================================================

WebServer::WebServer(DicomViewer* viewer, QObject* parent)
    : QObject(parent)
    , m_server(nullptr)
    , m_viewer(viewer)
    , m_port(0)
    , m_useSSL(false)
    , m_currentWindow(256.0)  // Default window (matching VolumeRenderer default)
    , m_currentLevel(128.0)   // Default level (matching VolumeRenderer default)
{
}

WebServer::~WebServer() {
    stop();
}

bool WebServer::start(quint16 port, bool useSSL) {
    if (m_server && m_server->isListening()) {
        qWarning() << "Web server is already running on port" << m_port;
        return false;
    }

    m_useSSL = useSSL;

    // Create appropriate server type
    if (m_useSSL) {
        // Check SSL support
        if (!QSslSocket::supportsSsl()) {
            qCritical() << "SSL is not supported on this system!";
            qCritical() << "Qt SSL version:" << QSslSocket::sslLibraryBuildVersionString();
            qCritical() << "Please install OpenSSL libraries";
            return false;
        }

        // Load SSL certificates
        if (!loadSSLCertificates()) {
            qCritical() << "Failed to load SSL certificates";
            qCritical() << "Please run: ./scripts/generate_ssl_cert.sh";
            return false;
        }

        qInfo() << "SSL/TLS enabled";
        qInfo() << "Certificate loaded successfully";

        // Create SSL server
        SslServer* sslServer = new SslServer(this, this);
        sslServer->setSSLConfiguration(m_sslCertificate, m_sslPrivateKey);
        m_server = sslServer;
    } else {
        // Create regular TCP server
        m_server = new QTcpServer(this);
        connect(m_server, &QTcpServer::newConnection, this, &WebServer::onNewConnection);
    }

    if (!m_server->listen(QHostAddress::Any, port)) {
        qCritical() << "Failed to start web server:" << m_server->errorString();
        return false;
    }

    m_port = m_server->serverPort();
    QString protocol = m_useSSL ? "https" : "http";
    qInfo() << "Web server started on port" << m_port << (m_useSSL ? "(HTTPS)" : "(HTTP)");
    qInfo() << "Access the web interface at:" << protocol << "://localhost:" << m_port;
    emit serverStarted(m_port);
    return true;
}

void WebServer::stop() {
    if (!m_server || !m_server->isListening()) {
        return;
    }

    m_server->close();
    m_server->deleteLater();
    m_server = nullptr;
    m_port = 0;
    qInfo() << "Web server stopped";
    emit serverStopped();
}

bool WebServer::isRunning() const {
    return m_server && m_server->isListening();
}

quint16 WebServer::port() const {
    return m_port;
}

bool WebServer::isSSLEnabled() const {
    return m_useSSL;
}

void WebServer::onNewConnection() {
    // This is only called for non-SSL connections
    while (m_server->hasPendingConnections()) {
        QTcpSocket* socket = m_server->nextPendingConnection();

        qDebug() << "=== New HTTP connection received ===";
        qDebug() << "Client address:" << socket->peerAddress().toString();

        // Connect signals for regular HTTP
        connect(socket, &QTcpSocket::readyRead, this, &WebServer::onReadyRead);
        connect(socket, &QTcpSocket::disconnected, this, &WebServer::onDisconnected);

        QString clientAddress = socket->peerAddress().toString();
        qDebug() << "HTTP client connected from" << clientAddress;
        emit clientConnected(clientAddress);
        qDebug() << "=== Connection setup complete ===";
    }
}

void WebServer::handleNewSslSocket(QSslSocket* sslSocket) {
    if (!sslSocket) return;

    qDebug() << "=== Handling encrypted SSL socket ===";
    qDebug() << "Client address:" << sslSocket->peerAddress().toString();

    // Connect data signals now that encryption is established
    connect(sslSocket, &QTcpSocket::readyRead, this, &WebServer::onReadyRead);
    connect(sslSocket, &QTcpSocket::disconnected, this, &WebServer::onDisconnected);

    QString clientAddress = sslSocket->peerAddress().toString();
    emit clientConnected(clientAddress);
    qDebug() << "=== SSL Connection ready for data transfer ===";
}

void WebServer::onReadyRead() {
    QTcpSocket* socket = qobject_cast<QTcpSocket*>(sender());
    if (!socket) return;

    // Read all available data
    QByteArray newData = socket->readAll();

    // Accumulate data (protected by mutex)
    m_socketBuffersMutex.lock();
    m_socketBuffers[socket].append(newData);

    // Check if we have a complete HTTP request (ends with \r\n\r\n for headers)
    QByteArray buffer = m_socketBuffers[socket];
    m_socketBuffersMutex.unlock();

    int headerEnd = buffer.indexOf("\r\n\r\n");
    if (headerEnd == -1) {
        // Not complete yet
        return;
    }

    // Parse and handle request
    HttpRequest request = parseRequest(buffer);
    emit requestReceived(request.method, request.path);

    HttpResponse response = handleRequest(request);
    sendResponse(socket, response);

    // Clear buffer and close connection (protected by mutex)
    m_socketBuffersMutex.lock();
    m_socketBuffers.remove(socket);
    m_socketBuffersMutex.unlock();

    socket->disconnectFromHost();
}

void WebServer::onDisconnected() {
    QTcpSocket* socket = qobject_cast<QTcpSocket*>(sender());
    if (socket) {
        qDebug() << "=== Client disconnected ===";
        qDebug() << "Client address:" << socket->peerAddress().toString();

        // Check if it's an SSL socket
        QSslSocket* sslSocket = qobject_cast<QSslSocket*>(socket);
        if (sslSocket) {
            qDebug() << "SSL socket - was encrypted:" << sslSocket->isEncrypted();
            qDebug() << "SSL socket state:" << sslSocket->state();
            if (sslSocket->error() != QAbstractSocket::UnknownSocketError) {
                qWarning() << "Socket had error:" << sslSocket->errorString();
            }
        }

        // Remove socket buffer (protected by mutex)
        m_socketBuffersMutex.lock();
        m_socketBuffers.remove(socket);
        m_socketBuffersMutex.unlock();

        socket->deleteLater();
    }
}

WebServer::HttpRequest WebServer::parseRequest(const QByteArray& data) {
    HttpRequest request;

    // Split into lines
    QList<QByteArray> lines = data.split('\n');
    if (lines.isEmpty()) return request;

    // Parse request line (e.g., "GET /api/patients HTTP/1.1")
    QList<QByteArray> requestLine = lines[0].trimmed().split(' ');
    if (requestLine.size() >= 3) {
        request.method = QString::fromLatin1(requestLine[0]);
        request.path = QString::fromLatin1(requestLine[1]);
        request.version = QString::fromLatin1(requestLine[2]);
    }

    // Parse headers
    int i = 1;
    for (; i < lines.size(); ++i) {
        QByteArray line = lines[i].trimmed();
        if (line.isEmpty()) {
            ++i;
            break;
        }

        int colonIndex = line.indexOf(':');
        if (colonIndex > 0) {
            QString key = QString::fromLatin1(line.left(colonIndex).trimmed());
            QString value = QString::fromLatin1(line.mid(colonIndex + 1).trimmed());
            request.headers[key] = value;
        }
    }

    // Parse body (if any)
    if (i < lines.size()) {
        for (; i < lines.size(); ++i) {
            request.body.append(lines[i]);
            if (i < lines.size() - 1) {
                request.body.append('\n');
            }
        }
    }

    return request;
}

WebServer::HttpResponse WebServer::handleRequest(const HttpRequest& request) {
    qDebug() << "Request:" << request.method << request.path;

    // Handle API requests
    if (request.path.startsWith("/api/")) {
        return handleApiRequest(request);
    }

    // Serve static files
    return serveStaticFile(request.path);
}

void WebServer::sendResponse(QTcpSocket* socket, const HttpResponse& response) {
    if (!socket || !socket->isValid()) return;

    // Build response
    QByteArray responseData;
    responseData.append("HTTP/1.1 ");
    responseData.append(QByteArray::number(response.statusCode));
    responseData.append(" ");
    responseData.append(response.statusMessage.toLatin1());
    responseData.append("\r\n");

    // Add headers
    for (auto it = response.headers.begin(); it != response.headers.end(); ++it) {
        responseData.append(it.key().toLatin1());
        responseData.append(": ");
        responseData.append(it.value().toLatin1());
        responseData.append("\r\n");
    }

    // Add standard headers
    responseData.append("Server: ShioRIS3-WebServer/1.0\r\n");
    responseData.append("Date: ");
    responseData.append(QDateTime::currentDateTimeUtc().toString(Qt::RFC2822Date).toLatin1());
    responseData.append("\r\n");
    responseData.append("Connection: close\r\n");
    responseData.append("Content-Length: ");
    responseData.append(QByteArray::number(response.body.size()));
    responseData.append("\r\n\r\n");
    responseData.append(response.body);

    // Send response
    socket->write(responseData);
    socket->flush();
}

WebServer::HttpResponse WebServer::serveStaticFile(const QString& path) {
    HttpResponse response;

    // Determine file path
    QString filePath;
    QString cleanPath = path;

    // Remove query parameters (e.g., ?v=3)
    int queryPos = cleanPath.indexOf('?');
    if (queryPos != -1) {
        cleanPath = cleanPath.left(queryPos);
    }

    if (cleanPath == "/" || cleanPath.isEmpty()) {
        filePath = "index.html";
    } else {
        filePath = cleanPath;
        if (filePath.startsWith("/")) {
            filePath = filePath.mid(1);
        }
    }

    // Security: prevent directory traversal
    if (filePath.contains("..")) {
        response.statusCode = 403;
        response.statusMessage = "Forbidden";
        response.body = "403 Forbidden";
        return response;
    }

    // Build full path to web_client directory
    QString webClientDir = QCoreApplication::applicationDirPath() + "/../web_client";
    QDir dir(webClientDir);
    QString fullPath = dir.absoluteFilePath(filePath);

    qDebug() << "Serving file:" << fullPath;

    // Try to open file
    QFile file(fullPath);
    if (!file.exists()) {
        response.statusCode = 404;
        response.statusMessage = "Not Found";
        response.body = "404 Not Found: " + filePath.toLatin1();
        return response;
    }

    if (!file.open(QIODevice::ReadOnly)) {
        response.statusCode = 500;
        response.statusMessage = "Internal Server Error";
        response.body = "500 Internal Server Error: Cannot read file";
        return response;
    }

    // Read file
    response.body = file.readAll();
    response.headers["Content-Type"] = getMimeType(filePath);
    response.headers["Cache-Control"] = "no-cache";

    return response;
}

WebServer::HttpResponse WebServer::handleApiRequest(const HttpRequest& request) {
    HttpResponse response;
    response.headers["Content-Type"] = "application/json";
    response.headers["Access-Control-Allow-Origin"] = "*"; // Enable CORS

    // Extract query parameters
    QString fullPath = request.path;
    QString path = fullPath;
    QMap<QString, QString> queryParams;

    int queryPos = fullPath.indexOf('?');
    if (queryPos != -1) {
        path = fullPath.left(queryPos);
        QString queryString = fullPath.mid(queryPos + 1);

        // Parse query parameters
        QStringList pairs = queryString.split('&');
        for (const QString& pair : pairs) {
            QStringList keyValue = pair.split('=');
            if (keyValue.size() == 2) {
                queryParams[keyValue[0]] = keyValue[1];
            }
        }
    }

    // Parse API path
    if (path == "/api/patients") {
        return apiGetPatients();
    } else if (path == "/api/window-level") {
        return apiGetWindowLevel();
    } else if (path.startsWith("/api/volume/")) {
        QString patientId = path.mid(12); // Remove "/api/volume/"
        return apiGetVolume(patientId);
    } else if (path.startsWith("/api/structures/")) {
        QString patientId = path.mid(16); // Remove "/api/structures/"
        return apiGetStructures(patientId, queryParams);
    } else if (path.startsWith("/api/dose/")) {
        QString patientId = path.mid(10); // Remove "/api/dose/"
        return apiGetDose(patientId);
    } else if (path.startsWith("/api/slice/")) {
        // Parse /api/slice/[patientId]/[orientation]/[index]
        QString remainder = path.mid(11); // Remove "/api/slice/"
        QStringList parts = remainder.split('/');
        if (parts.size() >= 3) {
            QString patientId = parts[0];
            QString orientation = parts[1];
            bool conversionOk = false;
            int sliceIndex = parts[2].toInt(&conversionOk);

            qDebug() << "Parsed slice request - patientId:" << patientId
                     << "orientation:" << orientation
                     << "sliceIndex:" << sliceIndex
                     << "conversion:" << (conversionOk ? "OK" : "FAILED");

            return apiGetSlice(patientId, orientation, sliceIndex, queryParams);
        }
    }

    // Unknown endpoint
    response.statusCode = 404;
    response.statusMessage = "Not Found";
    QJsonObject errorObj;
    errorObj["error"] = "Unknown API endpoint";
    errorObj["path"] = path;
    response.body = QJsonDocument(errorObj).toJson();

    return response;
}

QString WebServer::getMimeType(const QString& path) const {
    QString extension = QFileInfo(path).suffix().toLower();

    static QMap<QString, QString> mimeTypes = {
        {"html", "text/html; charset=utf-8"},
        {"htm", "text/html; charset=utf-8"},
        {"css", "text/css; charset=utf-8"},
        {"js", "application/javascript; charset=utf-8"},
        {"json", "application/json; charset=utf-8"},
        {"png", "image/png"},
        {"jpg", "image/jpeg"},
        {"jpeg", "image/jpeg"},
        {"gif", "image/gif"},
        {"svg", "image/svg+xml"},
        {"ico", "image/x-icon"},
        {"wasm", "application/wasm"},
        {"glb", "model/gltf-binary"},
        {"gltf", "model/gltf+json"}
    };

    return mimeTypes.value(extension, "application/octet-stream");
}

// API endpoint implementations
WebServer::HttpResponse WebServer::apiGetPatients() {
    HttpResponse response;

    QJsonObject resultObj;
    resultObj["status"] = "success";

    QJsonArray patientsArray;

    // Check if a volume is currently loaded
    if (m_viewer && m_viewer->isVolumeLoaded()) {
        QJsonObject currentPatient;
        currentPatient["id"] = "current";
        currentPatient["name"] = "Currently Loaded Volume";
        currentPatient["studyDate"] = QDate::currentDate().toString(Qt::ISODate);

        const DicomVolume& volume = m_viewer->getVolume();
        currentPatient["width"] = volume.width();
        currentPatient["height"] = volume.height();
        currentPatient["depth"] = volume.depth();

        patientsArray.append(currentPatient);
    }

    resultObj["patients"] = patientsArray;
    resultObj["count"] = patientsArray.size();
    resultObj["message"] = patientsArray.isEmpty() ?
        "No volume loaded. Please load a DICOM volume in ShioRIS3." :
        "Current loaded volume available";

    response.body = QJsonDocument(resultObj).toJson();
    return response;
}

WebServer::HttpResponse WebServer::apiGetVolume(const QString& patientId) {
    HttpResponse response;

    QJsonObject resultObj;
    resultObj["patientId"] = patientId;

    // Check if volume is loaded
    if (!m_viewer || !m_viewer->isVolumeLoaded()) {
        response.statusCode = 404;
        response.statusMessage = "Not Found";
        resultObj["status"] = "error";
        resultObj["message"] = "No volume loaded. Please load a DICOM volume in ShioRIS3.";
        response.body = QJsonDocument(resultObj).toJson();
        return response;
    }

    const DicomVolume& volume = m_viewer->getVolume();

    resultObj["status"] = "success";

    // Volume metadata
    QJsonObject volumeInfo;
    volumeInfo["width"] = volume.width();
    volumeInfo["height"] = volume.height();
    volumeInfo["depth"] = volume.depth();
    volumeInfo["spacingX"] = volume.spacingX();
    volumeInfo["spacingY"] = volume.spacingY();
    volumeInfo["spacingZ"] = volume.spacingZ();
    volumeInfo["originX"] = volume.originX();
    volumeInfo["originY"] = volume.originY();
    volumeInfo["originZ"] = volume.originZ();
    volumeInfo["frameOfReferenceUID"] = volume.frameOfReferenceUID();

    resultObj["volume"] = volumeInfo;
    resultObj["message"] = "Volume metadata retrieved successfully";

    response.body = QJsonDocument(resultObj).toJson();
    return response;
}

WebServer::HttpResponse WebServer::apiGetStructures(const QString& patientId, const QMap<QString, QString>& queryParams) {
    HttpResponse response;

    QJsonObject resultObj;
    resultObj["patientId"] = patientId;

    // Parse simplification parameter (default: 1 = no simplification)
    int simplify = 1;
    if (queryParams.contains("simplify")) {
        bool ok = false;
        int value = queryParams["simplify"].toInt(&ok);
        if (ok && value >= 1 && value <= 10) {
            simplify = value;
        }
    }

    // Check if RT Structure is loaded
    if (!m_viewer || !m_viewer->isRTStructLoaded()) {
        response.statusCode = 404;
        response.statusMessage = "Not Found";
        resultObj["status"] = "error";
        resultObj["message"] = "No RT Structure loaded. Please load an RT Structure file in ShioRIS3.";
        response.body = QJsonDocument(resultObj).toJson();
        return response;
    }

    // Also need volume for coordinate transformation
    if (!m_viewer->isVolumeLoaded()) {
        response.statusCode = 404;
        response.statusMessage = "Not Found";
        resultObj["status"] = "error";
        resultObj["message"] = "No volume loaded. Volume is required for RT Structure visualization.";
        response.body = QJsonDocument(resultObj).toJson();
        return response;
    }

    const RTStructureSet& rtstruct = m_viewer->getRTStruct();
    const DicomVolume& volume = m_viewer->getVolume();

    resultObj["status"] = "success";
    resultObj["simplification"] = simplify;

    QJsonArray structuresArray;
    int totalContours = 0;
    int totalPoints = 0;
    int originalPoints = 0;

    // Get contours for each visible ROI
    int roiCount = rtstruct.roiCount();
    for (int roiIndex = 0; roiIndex < roiCount; ++roiIndex) {
        if (!rtstruct.isROIVisible(roiIndex)) {
            continue; // Skip invisible ROIs
        }

        // Get contours for this specific ROI only
        StructureLine3DList roiContours = rtstruct.roiContours3D(roiIndex, volume);

        QJsonObject roiObj;
        roiObj["name"] = rtstruct.roiName(roiIndex);
        roiObj["index"] = roiIndex;
        roiObj["visible"] = true;

        // Convert ROI contours to JSON
        QJsonArray contoursArray;
        for (const StructureLine3D& line : roiContours) {
            QJsonObject contourObj;

            // Color
            QJsonObject colorObj;
            colorObj["r"] = line.color.red();
            colorObj["g"] = line.color.green();
            colorObj["b"] = line.color.blue();
            colorObj["a"] = line.color.alpha();
            contourObj["color"] = colorObj;

            // Points - apply simplification (downsample by taking every Nth point)
            QJsonArray pointsArray;
            originalPoints += line.points.size();

            if (simplify == 1) {
                // No simplification - send all points
                for (const QVector3D& point : line.points) {
                    QJsonObject pointObj;
                    pointObj["x"] = point.x();
                    pointObj["y"] = point.y();
                    pointObj["z"] = point.z();
                    pointsArray.append(pointObj);
                }
                totalPoints += line.points.size();
            } else {
                // Downsample: keep first point, every Nth point, and last point
                int numPoints = line.points.size();
                if (numPoints > 0) {
                    // Always include first point
                    QJsonObject firstPoint;
                    firstPoint["x"] = line.points[0].x();
                    firstPoint["y"] = line.points[0].y();
                    firstPoint["z"] = line.points[0].z();
                    pointsArray.append(firstPoint);
                    totalPoints++;

                    // Include every Nth point
                    for (int i = simplify; i < numPoints - 1; i += simplify) {
                        QJsonObject pointObj;
                        pointObj["x"] = line.points[i].x();
                        pointObj["y"] = line.points[i].y();
                        pointObj["z"] = line.points[i].z();
                        pointsArray.append(pointObj);
                        totalPoints++;
                    }

                    // Always include last point (if different from first)
                    if (numPoints > 1) {
                        QJsonObject lastPoint;
                        lastPoint["x"] = line.points[numPoints - 1].x();
                        lastPoint["y"] = line.points[numPoints - 1].y();
                        lastPoint["z"] = line.points[numPoints - 1].z();
                        pointsArray.append(lastPoint);
                        totalPoints++;
                    }
                }
            }
            contourObj["points"] = pointsArray;

            contoursArray.append(contourObj);
        }

        roiObj["contours"] = contoursArray;
        roiObj["contourCount"] = contoursArray.size();
        totalContours += contoursArray.size();

        structuresArray.append(roiObj);
    }

    resultObj["structures"] = structuresArray;
    resultObj["structureCount"] = structuresArray.size();
    resultObj["totalContours"] = totalContours;
    resultObj["totalPoints"] = totalPoints;
    resultObj["originalPoints"] = originalPoints;

    double reductionPercent = (originalPoints > 0)
        ? ((originalPoints - totalPoints) * 100.0 / originalPoints)
        : 0.0;

    resultObj["message"] = QString("RT Structure data retrieved successfully (simplify=%1, %2 -> %3 points, %4% reduction)")
                              .arg(simplify)
                              .arg(originalPoints)
                              .arg(totalPoints)
                              .arg(reductionPercent, 0, 'f', 1);

    response.body = QJsonDocument(resultObj).toJson();

    // Log response size for debugging
    qDebug() << "apiGetStructures:" << structuresArray.size() << "structures with"
             << totalContours << "contours," << totalPoints << "points (original:"
             << originalPoints << ", reduction:" << reductionPercent << "%), JSON size:"
             << (response.body.size() / 1024.0) << "KB";

    return response;
}

WebServer::HttpResponse WebServer::apiGetDose(const QString& patientId) {
    HttpResponse response;
    response.headers["Content-Type"] = "application/json";

    QJsonObject resultObj;
    resultObj["patientId"] = patientId;

    // Check if dose isosurfaces are available
    if (!m_viewer) {
        resultObj["status"] = "error";
        resultObj["message"] = "Viewer not available";
        response.body = QJsonDocument(resultObj).toJson();
        return response;
    }

    const QVector<DoseIsosurface>& isosurfaces = m_viewer->getDoseIsosurfaces();

    if (isosurfaces.isEmpty()) {
        resultObj["status"] = "success";
        resultObj["message"] = "No dose isosurfaces available. Please generate 3D Isosurface in ShioRIS3.";
        resultObj["isosurfaceCount"] = 0;
        resultObj["isosurfaces"] = QJsonArray();
        response.body = QJsonDocument(resultObj).toJson();
        return response;
    }

    // Convert isosurfaces to JSON
    QJsonArray isosurfacesArray;
    int totalTriangles = 0;

    for (const DoseIsosurface& surface : isosurfaces) {
        if (surface.isEmpty()) continue;

        QJsonObject surfaceObj;

        // Color and opacity
        QColor color = surface.color();
        QJsonObject colorObj;
        colorObj["r"] = color.red();
        colorObj["g"] = color.green();
        colorObj["b"] = color.blue();
        surfaceObj["color"] = colorObj;
        surfaceObj["opacity"] = surface.opacity();

        // Triangles - use flat arrays for efficiency
        const QVector<DoseTriangle>& triangles = surface.triangles();

        // Vertices: flat array [x0,y0,z0, x1,y1,z1, x2,y2,z2, ...] (9 floats per triangle)
        QJsonArray verticesArray;
        // Normals: flat array [nx0,ny0,nz0, nx1,ny1,nz1, ...] (3 floats per triangle)
        QJsonArray normalsArray;

        for (const DoseTriangle& tri : triangles) {
            // Add 3 vertices (9 coordinates)
            for (int i = 0; i < 3; ++i) {
                verticesArray.append(tri.vertices[i].x());
                verticesArray.append(tri.vertices[i].y());
                verticesArray.append(tri.vertices[i].z());
            }

            // Add 1 normal (3 components)
            normalsArray.append(tri.normal.x());
            normalsArray.append(tri.normal.y());
            normalsArray.append(tri.normal.z());
        }

        surfaceObj["vertices"] = verticesArray;  // Flat array of vertex coordinates
        surfaceObj["normals"] = normalsArray;    // Flat array of normal vectors
        surfaceObj["triangleCount"] = triangles.size();
        totalTriangles += triangles.size();

        isosurfacesArray.append(surfaceObj);
    }

    resultObj["status"] = "success";
    resultObj["message"] = "Dose isosurfaces retrieved successfully";
    resultObj["isosurfaceCount"] = isosurfacesArray.size();
    resultObj["totalTriangles"] = totalTriangles;
    resultObj["isosurfaces"] = isosurfacesArray;

    response.body = QJsonDocument(resultObj).toJson();

    // Log response size for debugging
    qDebug() << "apiGetDose: Sending" << isosurfacesArray.size() << "isosurfaces with"
             << totalTriangles << "triangles, JSON size:" << (response.body.size() / 1024.0) << "KB";

    return response;
}

WebServer::HttpResponse WebServer::apiGetSlice(const QString& patientId, const QString& orientation, int sliceIndex, const QMap<QString, QString>& queryParams) {
    HttpResponse response;

    // Check if volume is loaded
    if (!m_viewer || !m_viewer->isVolumeLoaded()) {
        response.statusCode = 404;
        response.statusMessage = "Not Found";
        response.headers["Content-Type"] = "application/json";
        QJsonObject errorObj;
        errorObj["status"] = "error";
        errorObj["message"] = "No volume loaded";
        response.body = QJsonDocument(errorObj).toJson();
        return response;
    }

    const DicomVolume& volume = m_viewer->getVolume();

    // Parse orientation
    DicomVolume::Orientation ori;
    if (orientation == "axial") {
        ori = DicomVolume::Orientation::Axial;
    } else if (orientation == "sagittal") {
        ori = DicomVolume::Orientation::Sagittal;
    } else if (orientation == "coronal") {
        ori = DicomVolume::Orientation::Coronal;
    } else {
        response.statusCode = 400;
        response.statusMessage = "Bad Request";
        response.headers["Content-Type"] = "application/json";
        QJsonObject errorObj;
        errorObj["status"] = "error";
        errorObj["message"] = "Invalid orientation. Use: axial, sagittal, or coronal";
        response.body = QJsonDocument(errorObj).toJson();
        return response;
    }

    // Validate slice index
    int maxSlice = 0;
    if (ori == DicomVolume::Orientation::Axial) {
        maxSlice = volume.depth() - 1;
    } else if (ori == DicomVolume::Orientation::Sagittal) {
        maxSlice = volume.width() - 1;
    } else if (ori == DicomVolume::Orientation::Coronal) {
        maxSlice = volume.height() - 1;
    }

    if (sliceIndex < 0 || sliceIndex > maxSlice) {
        response.statusCode = 400;
        response.statusMessage = "Bad Request";
        response.headers["Content-Type"] = "application/json";
        QJsonObject errorObj;
        errorObj["status"] = "error";
        errorObj["message"] = QString("Slice index out of range. Valid range: 0-%1").arg(maxSlice);
        response.body = QJsonDocument(errorObj).toJson();
        return response;
    }

    // Calculate and log position
    double position = (maxSlice > 0) ? (double)sliceIndex / maxSlice : 0.0;
    qInfo() << QString("Updating %1 slice: position=%2, index=%3/%4")
               .arg(orientation)
               .arg(position, 0, 'f', 3)
               .arg(sliceIndex)
               .arg(maxSlice);

    // Get slice image with window/level from query parameters or defaults
    double window = 256.0;  // Default
    double level = 128.0;   // Default

    // Override with query parameters if provided
    if (queryParams.contains("window")) {
        bool ok = false;
        double w = queryParams["window"].toDouble(&ok);
        if (ok && w > 0) {
            window = w;
        }
    }

    if (queryParams.contains("level")) {
        bool ok = false;
        double l = queryParams["level"].toDouble(&ok);
        if (ok) {
            level = l;
        }
    }

    qDebug() << "Slice request - window:" << window << "level:" << level;

    QImage sliceImage = volume.getSlice(sliceIndex, ori, window, level);

    // Flip Axial images vertically to flip Y-axis (DICOM anterior-posterior direction)
    if (ori == DicomVolume::Orientation::Axial) {
        sliceImage = sliceImage.mirrored(true, false);  // true=flip X axis, false=no Y flip
        qDebug() << "Axial slice flipped Y axis (DICOM AP direction)";
    }
    // Flip Sagittal images vertically only to correct AP direction (Y-axis in DICOM)
    else if (ori == DicomVolume::Orientation::Sagittal) {
        sliceImage = sliceImage.mirrored(false, true);  // false=no horizontal flip, true=vertical flip (AP direction)
        qDebug() << "Sagittal slice flipped vertically (AP direction correction)";
    }

    if (sliceImage.isNull()) {
        response.statusCode = 500;
        response.statusMessage = "Internal Server Error";
        response.headers["Content-Type"] = "application/json";
        QJsonObject errorObj;
        errorObj["status"] = "error";
        errorObj["message"] = "Failed to generate slice image";
        response.body = QJsonDocument(errorObj).toJson();
        return response;
    }

    // Apply brightness-based alpha transparency
    // Convert to RGBA format to manipulate alpha channel
    QImage transparentImage = sliceImage.convertToFormat(QImage::Format_RGBA8888);

    // Use pixel brightness as alpha value to match Desktop OpenGL rendering
    // This allows transparency based on CT values, matching the visual appearance in MainWindow
    for (int y = 0; y < transparentImage.height(); ++y) {
        QRgb* line = reinterpret_cast<QRgb*>(transparentImage.scanLine(y));
        for (int x = 0; x < transparentImage.width(); ++x) {
            QRgb pixel = line[x];
            int r = qRed(pixel);
            int g = qGreen(pixel);
            int b = qBlue(pixel);
            int brightness = qGray(pixel);  // Get grayscale brightness (0-255)
            // Use brightness as alpha to create smooth transparency gradient
            // Keep original RGB values, only modify alpha channel
            line[x] = qRgba(r, g, b, brightness);
        }
    }

    qDebug() << "Alpha channel set to match pixel brightness for VR display";

    // Convert QImage to PNG
    QBuffer buffer;
    buffer.open(QIODevice::WriteOnly);
    if (!transparentImage.save(&buffer, "PNG")) {
        response.statusCode = 500;
        response.statusMessage = "Internal Server Error";
        response.headers["Content-Type"] = "application/json";
        QJsonObject errorObj;
        errorObj["status"] = "error";
        errorObj["message"] = "Failed to encode image to PNG";
        response.body = QJsonDocument(errorObj).toJson();
        return response;
    }

    // Return PNG image
    response.headers["Content-Type"] = "image/png";
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"; // Disable caching for development
    response.headers["Pragma"] = "no-cache"; // HTTP 1.0 compatibility
    response.headers["Expires"] = "0"; // Proxies
    response.body = buffer.data();

    qDebug() << "Served slice:" << orientation << sliceIndex << "size:" << sliceImage.size();

    return response;
}

WebServer::HttpResponse WebServer::apiGetWindowLevel() {
    HttpResponse response;

    QMutexLocker locker(&m_windowLevelMutex);

    QJsonObject resultObj;
    resultObj["status"] = "success";
    resultObj["window"] = m_currentWindow;
    resultObj["level"] = m_currentLevel;

    response.headers["Content-Type"] = "application/json";
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate";
    response.body = QJsonDocument(resultObj).toJson();

    return response;
}

void WebServer::setWindowLevel(double window, double level) {
    QMutexLocker locker(&m_windowLevelMutex);
    m_currentWindow = window;
    m_currentLevel = level;
    qDebug() << "WebServer: Window/Level updated to w=" << window << ", l=" << level;
}

void WebServer::getWindowLevel(double& window, double& level) const {
    QMutexLocker locker(const_cast<QMutex*>(&m_windowLevelMutex));
    window = m_currentWindow;
    level = m_currentLevel;
}

bool WebServer::loadSSLCertificates() {
    // Determine certificate path - try multiple locations
    QStringList searchPaths;

    // 1. Try relative to application directory (development builds)
    searchPaths << QCoreApplication::applicationDirPath() + "/../ssl_certs";

    // 2. Try in app bundle Contents directory (macOS .app)
    searchPaths << QCoreApplication::applicationDirPath() + "/../../../ssl_certs";

    // 3. Try in project root (when running from build directory)
    searchPaths << QCoreApplication::applicationDirPath() + "/../../ssl_certs";

    // 4. Try absolute path to source directory (fallback)
    QString sourcePath = __FILE__;
    QFileInfo sourceInfo(sourcePath);
    searchPaths << sourceInfo.absoluteDir().absolutePath() + "/../../ssl_certs";

    QString certPath;
    QString keyPath;
    QString certDir;

    // Search for certificates in all possible locations
    for (const QString& searchPath : searchPaths) {
        QDir dir(searchPath);
        QString testCertPath = dir.absoluteFilePath("server.crt");
        QString testKeyPath = dir.absoluteFilePath("server.key");

        if (QFile::exists(testCertPath) && QFile::exists(testKeyPath)) {
            certPath = testCertPath;
            keyPath = testKeyPath;
            certDir = dir.absolutePath();
            break;
        }
    }

    if (certPath.isEmpty() || keyPath.isEmpty()) {
        qCritical() << "Could not find SSL certificates in any of the following locations:";
        for (const QString& searchPath : searchPaths) {
            QDir dir(searchPath);
            qCritical() << "  -" << dir.absolutePath();
        }
        qCritical() << "Please run: ./scripts/generate_ssl_cert.sh from the project root";
        return false;
    }

    qDebug() << "Found SSL certificates in:" << certDir;
    qDebug() << "Loading SSL certificate from:" << certPath;
    qDebug() << "Loading SSL private key from:" << keyPath;

    // Load certificate
    QFile certFile(certPath);
    if (!certFile.open(QIODevice::ReadOnly)) {
        qCritical() << "Failed to open certificate file:" << certPath;
        qCritical() << "Error:" << certFile.errorString();
        return false;
    }

    QList<QSslCertificate> certList = QSslCertificate::fromDevice(&certFile, QSsl::Pem);
    certFile.close();

    if (certList.isEmpty()) {
        qCritical() << "No valid certificates found in:" << certPath;
        return false;
    }

    m_sslCertificate = certList.first();

    // Load private key
    QFile keyFile(keyPath);
    if (!keyFile.open(QIODevice::ReadOnly)) {
        qCritical() << "Failed to open private key file:" << keyPath;
        qCritical() << "Error:" << keyFile.errorString();
        return false;
    }

    m_sslPrivateKey = QSslKey(&keyFile, QSsl::Rsa, QSsl::Pem, QSsl::PrivateKey);
    keyFile.close();

    if (m_sslPrivateKey.isNull()) {
        qCritical() << "Failed to load private key from:" << keyPath;
        return false;
    }

    qInfo() << "SSL certificates loaded successfully";
    qInfo() << "Certificate subject:" << m_sslCertificate.subjectDisplayName();
    qInfo() << "Certificate valid from:" << m_sslCertificate.effectiveDate().toString();
    qInfo() << "Certificate valid to:" << m_sslCertificate.expiryDate().toString();

    return true;
}

