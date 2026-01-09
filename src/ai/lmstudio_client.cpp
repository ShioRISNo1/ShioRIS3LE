#include "ai/lmstudio_client.h"

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QJsonValue>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QUrl>
#include <QDebug>
#include <QTimer>

LmStudioClient::LmStudioClient(QObject *parent)
    : QObject(parent),
      m_endpoint(QUrl(QStringLiteral("http://localhost:1234/v1/chat/completions"))),
      m_model(QStringLiteral("mistralai/magistral-small-2509")),
      m_timeoutMs(30000),        // 30 seconds default
      m_maxRetries(3),           // 3 retries default
      m_retryDelayMs(1000),      // 1 second base delay
      m_networkManager(nullptr),
      m_currentReply(nullptr),
      m_timeoutTimer(nullptr),
      m_requestTimedOut(false),
      m_lastTemperature(0.2),
      m_lastStreaming(false),
      m_currentRetryCount(0) {

  m_timeoutTimer = new QTimer(this);
  m_timeoutTimer->setSingleShot(true);
  connect(m_timeoutTimer, &QTimer::timeout, this, &LmStudioClient::handleTimeout);
}

void LmStudioClient::setEndpoint(const QUrl &endpoint) { m_endpoint = endpoint; }

void LmStudioClient::setModel(const QString &model) { m_model = model; }

void LmStudioClient::setApiKey(const QString &apiKey) { m_apiKey = apiKey; }

void LmStudioClient::setTimeout(int timeoutMs) {
  m_timeoutMs = qMax(1000, timeoutMs); // Minimum 1 second
}

void LmStudioClient::setMaxRetries(int maxRetries) {
  m_maxRetries = qMax(0, maxRetries); // Non-negative
}

void LmStudioClient::setRetryDelay(int delayMs) {
  m_retryDelayMs = qMax(100, delayMs); // Minimum 100ms
}

QNetworkAccessManager *LmStudioClient::networkManager() {
  if (!m_networkManager) {
    m_networkManager = new QNetworkAccessManager(this);
  }
  return m_networkManager;
}

QUrl LmStudioClient::modelsEndpoint() const {
  if (!m_endpoint.isValid())
    return QUrl();

  QUrl modelsUrl(m_endpoint);
  QString path = modelsUrl.path();
  const QString marker = QStringLiteral("/v1/");
  const int index = path.indexOf(marker);
  if (index >= 0) {
    path = path.left(index + marker.size());
  } else {
    path = QStringLiteral("/v1/");
  }
  if (!path.endsWith(QLatin1Char('/')))
    path.append(QLatin1Char('/'));
  path.append(QStringLiteral("models"));

  modelsUrl.setPath(path);
  modelsUrl.setQuery(QString());
  modelsUrl.setFragment(QString());
  return modelsUrl;
}

void LmStudioClient::fetchAvailableModels() {
  if (!m_endpoint.isValid() || m_endpoint.isEmpty()) {
    emit modelsFetchFailed(tr("LMStudioのエンドポイントURLが無効です。"));
    return;
  }

  const QUrl url = modelsEndpoint();
  if (!url.isValid()) {
    emit modelsFetchFailed(tr("モデル一覧の取得先URLを構築できませんでした。"));
    return;
  }

  QNetworkRequest request(url);
  request.setHeader(QNetworkRequest::ContentTypeHeader,
                    QStringLiteral("application/json"));
  if (!m_apiKey.trimmed().isEmpty()) {
    const QByteArray token = QByteArray("Bearer ") + m_apiKey.trimmed().toUtf8();
    request.setRawHeader("Authorization", token);
  }

  auto reply = networkManager()->get(request);

  connect(reply, &QNetworkReply::finished, this, [this, reply]() {
    reply->deleteLater();
    if (reply->error() != QNetworkReply::NoError) {
      emit modelsFetchFailed(reply->errorString());
      return;
    }

    const QByteArray raw = reply->readAll();
    QJsonParseError error;
    const QJsonDocument doc = QJsonDocument::fromJson(raw, &error);
    QStringList models;

    auto appendModel = [&models](const QString &value) {
      const QString trimmed = value.trimmed();
      if (!trimmed.isEmpty() && !models.contains(trimmed))
        models.append(trimmed);
    };

    if (error.error == QJsonParseError::NoError) {
      if (doc.isObject()) {
        const QJsonObject obj = doc.object();
        const QJsonArray data = obj.value(QStringLiteral("data")).toArray();
        for (const QJsonValue &item : data) {
          if (item.isObject()) {
            appendModel(item.toObject().value(QStringLiteral("id")).toString());
            appendModel(item.toObject().value(QStringLiteral("model")).toString());
            appendModel(item.toObject().value(QStringLiteral("name")).toString());
          } else if (item.isString()) {
            appendModel(item.toString());
          }
        }
        if (models.isEmpty()) {
          const QJsonArray explicitModels =
              obj.value(QStringLiteral("models")).toArray();
          for (const QJsonValue &item : explicitModels) {
            if (item.isObject()) {
              appendModel(
                  item.toObject().value(QStringLiteral("id")).toString());
              appendModel(
                  item.toObject().value(QStringLiteral("model")).toString());
              appendModel(
                  item.toObject().value(QStringLiteral("name")).toString());
            } else if (item.isString()) {
              appendModel(item.toString());
            }
          }
        }
      } else if (doc.isArray()) {
        const QJsonArray arr = doc.array();
        for (const QJsonValue &item : arr) {
          if (item.isObject()) {
            appendModel(item.toObject().value(QStringLiteral("id")).toString());
            appendModel(item.toObject().value(QStringLiteral("model")).toString());
            appendModel(item.toObject().value(QStringLiteral("name")).toString());
          } else if (item.isString()) {
            appendModel(item.toString());
          }
        }
      }
    }

    if (models.isEmpty()) {
      emit modelsFetchFailed(tr("モデル一覧の応答を解析できませんでした。"));
      return;
    }

    emit modelsUpdated(models);
  });

  connect(reply, &QNetworkReply::errorOccurred, this,
          [this](QNetworkReply::NetworkError) {});
}

void LmStudioClient::setupRequestTimeout(QNetworkReply *reply) {
  if (!reply || m_timeoutMs <= 0) {
    return;
  }

  m_timeoutTimer->setInterval(m_timeoutMs);
  m_timeoutTimer->start();
}

void LmStudioClient::handleTimeout() {
  if (m_currentReply) {
    qWarning() << "LmStudio request timed out after" << m_timeoutMs << "ms";
    m_requestTimedOut = true;
    m_currentReply->abort();

    if (m_lastStreaming) {
      emit streamFailed(tr("Request timed out after %1 seconds").arg(m_timeoutMs / 1000));
    } else {
      emit requestFailed(tr("Request timed out after %1 seconds").arg(m_timeoutMs / 1000));
    }
  }
}

void LmStudioClient::handleStreamData() {
  if (!m_currentReply) {
    return;
  }

  QByteArray data = m_currentReply->readAll();
  if (data.isEmpty()) {
    return;
  }

  // Reset timeout on data receipt
  if (m_timeoutTimer->isActive()) {
    m_timeoutTimer->start();
  }

  // Parse SSE (Server-Sent Events) format
  QString dataStr = QString::fromUtf8(data);
  QStringList lines = dataStr.split('\n');

  for (const QString &line : lines) {
    if (line.startsWith(QStringLiteral("data: "))) {
      QString jsonStr = line.mid(6).trimmed(); // Remove "data: " prefix

      if (jsonStr == QStringLiteral("[DONE]")) {
        continue;
      }

      QJsonParseError error;
      QJsonDocument doc = QJsonDocument::fromJson(jsonStr.toUtf8(), &error);

      if (error.error == QJsonParseError::NoError && doc.isObject()) {
        QJsonObject obj = doc.object();
        QJsonArray choices = obj.value(QStringLiteral("choices")).toArray();

        if (!choices.isEmpty()) {
          QJsonObject delta = choices.first().toObject().value(QStringLiteral("delta")).toObject();
          QString content = delta.value(QStringLiteral("content")).toString();

          if (!content.isEmpty()) {
            emit streamChunkReceived(content);
          }
        }
      }
    }
  }
}

bool LmStudioClient::shouldRetry(QNetworkReply::NetworkError error) const {
  // Retry on temporary network errors
  switch (error) {
    case QNetworkReply::ConnectionRefusedError:
    case QNetworkReply::RemoteHostClosedError:
    case QNetworkReply::HostNotFoundError:
    case QNetworkReply::TimeoutError:
    case QNetworkReply::TemporaryNetworkFailureError:
    case QNetworkReply::NetworkSessionFailedError:
    case QNetworkReply::UnknownNetworkError:
    case QNetworkReply::UnknownServerError:
    case QNetworkReply::ServiceUnavailableError:
      return true;
    default:
      return false;
  }
}

int LmStudioClient::calculateBackoffDelay(int retryCount) const {
  // Exponential backoff: baseDelay * 2^retryCount
  // Capped at 30 seconds
  int delay = m_retryDelayMs * (1 << retryCount);
  return qMin(delay, 30000);
}

void LmStudioClient::sendChatCompletion(const QString &systemPrompt,
                                        const QString &userPrompt,
                                        double temperature) {
  sendChatCompletionInternal(systemPrompt, userPrompt, temperature, false, 0);
}

void LmStudioClient::sendChatCompletionStream(const QString &systemPrompt,
                                               const QString &userPrompt,
                                               double temperature) {
  sendChatCompletionInternal(systemPrompt, userPrompt, temperature, true, 0);
}

void LmStudioClient::cancelCurrentRequest() {
  if (m_currentReply) {
    m_timeoutTimer->stop();
    m_currentReply->abort();
    m_currentReply = nullptr;
    emit requestCancelled();
  }
}

void LmStudioClient::checkConnection() {
  emit connectionCheckStarted();

  if (!m_endpoint.isValid() || m_endpoint.isEmpty()) {
    emit connectionCheckFailed(tr("Invalid endpoint URL"));
    return;
  }

  QUrl healthUrl = modelsEndpoint();
  QNetworkRequest request(healthUrl);
  request.setHeader(QNetworkRequest::ContentTypeHeader, QStringLiteral("application/json"));

  if (!m_apiKey.trimmed().isEmpty()) {
    const QByteArray token = QByteArray("Bearer ") + m_apiKey.trimmed().toUtf8();
    request.setRawHeader("Authorization", token);
  }

  auto reply = networkManager()->get(request);

  QTimer *connectionTimeout = new QTimer(this);
  connectionTimeout->setSingleShot(true);
  connectionTimeout->setInterval(5000); // 5 second timeout for connection check

  connect(connectionTimeout, &QTimer::timeout, this, [reply, this]() {
    reply->abort();
    emit connectionCheckFailed(tr("Connection check timed out"));
  });

  connect(reply, &QNetworkReply::finished, this, [this, reply, connectionTimeout]() {
    connectionTimeout->stop();
    connectionTimeout->deleteLater();
    reply->deleteLater();

    if (reply->error() == QNetworkReply::NoError) {
      emit connectionCheckSucceeded();
    } else {
      emit connectionCheckFailed(reply->errorString());
    }
  });

  connectionTimeout->start();
}

void LmStudioClient::sendChatCompletionInternal(const QString &systemPrompt,
                                                  const QString &userPrompt,
                                                  double temperature,
                                                  bool streaming,
                                                  int retryCount) {
  // Validation
  if (!m_endpoint.isValid() || m_endpoint.isEmpty()) {
    emit requestFailed(tr("LMStudioのエンドポイントURLが無効です。"));
    return;
  }
  if (m_model.trimmed().isEmpty()) {
    emit requestFailed(tr("モデル名を設定してください。"));
    return;
  }
  if (userPrompt.trimmed().isEmpty()) {
    emit requestFailed(tr("送信するプロンプトが空です。"));
    return;
  }

  // Store for potential retry
  m_lastSystemPrompt = systemPrompt;
  m_lastUserPrompt = userPrompt;
  m_lastTemperature = temperature;
  m_lastStreaming = streaming;
  m_currentRetryCount = retryCount;
  m_requestTimedOut = false;

  // Build request body
  QJsonObject body;
  body.insert(QStringLiteral("model"), m_model);

  QJsonArray messages;
  if (!systemPrompt.trimmed().isEmpty()) {
    QJsonObject sys;
    sys.insert(QStringLiteral("role"), QStringLiteral("system"));
    sys.insert(QStringLiteral("content"), systemPrompt);
    messages.append(sys);
  }
  QJsonObject user;
  user.insert(QStringLiteral("role"), QStringLiteral("user"));
  user.insert(QStringLiteral("content"), userPrompt);
  messages.append(user);

  body.insert(QStringLiteral("messages"), messages);
  body.insert(QStringLiteral("temperature"), temperature);
  body.insert(QStringLiteral("stream"), streaming);

  QNetworkRequest request(m_endpoint);
  request.setHeader(QNetworkRequest::ContentTypeHeader, QStringLiteral("application/json"));

  if (!m_apiKey.trimmed().isEmpty()) {
    const QByteArray token = QByteArray("Bearer ") + m_apiKey.trimmed().toUtf8();
    request.setRawHeader("Authorization", token);
  }

  // Send request
  m_currentReply = networkManager()->post(request, QJsonDocument(body).toJson());

  // Setup timeout
  setupRequestTimeout(m_currentReply);

  emit requestStarted();

  if (streaming) {
    // Handle streaming response
    QNetworkReply *reply = m_currentReply;

    connect(reply, &QNetworkReply::readyRead, this, &LmStudioClient::handleStreamData);

    connect(reply, &QNetworkReply::finished, this, [this, reply]() {
      m_timeoutTimer->stop();

      if (m_requestTimedOut) {
        reply->deleteLater();
        m_currentReply = nullptr;
        return;
      }

      if (reply->error() == QNetworkReply::NoError) {
        emit streamFinished();
      } else {
        if (shouldRetry(reply->error()) && m_currentRetryCount < m_maxRetries) {
          qDebug() << "Stream request failed, retrying..." << (m_currentRetryCount + 1) << "of" << m_maxRetries;
          emit requestRetrying(m_currentRetryCount + 1, m_maxRetries);

          int delay = calculateBackoffDelay(m_currentRetryCount);
          QTimer::singleShot(delay, this, [this]() {
            sendChatCompletionInternal(m_lastSystemPrompt, m_lastUserPrompt,
                                        m_lastTemperature, m_lastStreaming,
                                        m_currentRetryCount + 1);
          });
        } else {
          emit streamFailed(reply->errorString());
        }
      }

      reply->deleteLater();
      m_currentReply = nullptr;
    });
  } else {
    // Handle normal response
    QNetworkReply *reply = m_currentReply;

    connect(reply, &QNetworkReply::finished, this, [this, reply]() {
      m_timeoutTimer->stop();

      if (m_requestTimedOut) {
        reply->deleteLater();
        m_currentReply = nullptr;
        return;
      }

      if (reply->error() != QNetworkReply::NoError) {
        if (shouldRetry(reply->error()) && m_currentRetryCount < m_maxRetries) {
          qDebug() << "Request failed, retrying..." << (m_currentRetryCount + 1) << "of" << m_maxRetries;
          emit requestRetrying(m_currentRetryCount + 1, m_maxRetries);

          int delay = calculateBackoffDelay(m_currentRetryCount);
          QTimer::singleShot(delay, this, [this]() {
            sendChatCompletionInternal(m_lastSystemPrompt, m_lastUserPrompt,
                                        m_lastTemperature, m_lastStreaming,
                                        m_currentRetryCount + 1);
          });
          reply->deleteLater();
          m_currentReply = nullptr;
          return;
        } else {
          emit requestFailed(reply->errorString());
          reply->deleteLater();
          m_currentReply = nullptr;
          return;
        }
      }

      const QByteArray raw = reply->readAll();
      QJsonParseError error;
      QJsonDocument doc = QJsonDocument::fromJson(raw, &error);
      QString content;

      if (error.error == QJsonParseError::NoError && doc.isObject()) {
        const QJsonObject obj = doc.object();
        const QJsonArray choices = obj.value(QStringLiteral("choices")).toArray();
        if (!choices.isEmpty()) {
          const QJsonObject message =
              choices.first().toObject().value(QStringLiteral("message")).toObject();
          content = message.value(QStringLiteral("content")).toString();
        }
        if (content.isEmpty()) {
          content = QString::fromUtf8(raw);
        }
      } else {
        content = QString::fromUtf8(raw);
      }

      emit requestFinished(content);
      reply->deleteLater();
      m_currentReply = nullptr;
    });
  }
}
