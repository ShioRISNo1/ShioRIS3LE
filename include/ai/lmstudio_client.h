#ifndef LMSTUDIO_CLIENT_H
#define LMSTUDIO_CLIENT_H

#include <QObject>
#include <QString>
#include <QStringList>
#include <QUrl>
#include <QTimer>
#include <QNetworkReply>

class QNetworkAccessManager;

class LmStudioClient : public QObject {
  Q_OBJECT

public:
  explicit LmStudioClient(QObject *parent = nullptr);

  // Configuration methods
  void setEndpoint(const QUrl &endpoint);
  void setModel(const QString &model);
  void setApiKey(const QString &apiKey);
  void setTimeout(int timeoutMs);
  void setMaxRetries(int maxRetries);
  void setRetryDelay(int delayMs);

  // Getters
  QUrl endpoint() const { return m_endpoint; }
  QString model() const { return m_model; }
  QString apiKey() const { return m_apiKey; }
  int timeout() const { return m_timeoutMs; }
  int maxRetries() const { return m_maxRetries; }
  int retryDelay() const { return m_retryDelayMs; }

  // Request methods
  void sendChatCompletion(const QString &systemPrompt,
                          const QString &userPrompt,
                          double temperature = 0.2);
  void sendChatCompletionStream(const QString &systemPrompt,
                                const QString &userPrompt,
                                double temperature = 0.2);
  void fetchAvailableModels();
  void cancelCurrentRequest();

  // Connection testing
  void checkConnection();

signals:
  // Request lifecycle signals
  void requestStarted();
  void requestFinished(const QString &responseText);
  void requestFailed(const QString &errorMessage);
  void requestCancelled();
  void requestRetrying(int attemptNumber, int maxAttempts);

  // Streaming signals
  void streamChunkReceived(const QString &chunk);
  void streamFinished();
  void streamFailed(const QString &errorMessage);

  // Model management signals
  void modelsUpdated(const QStringList &models);
  void modelsFetchFailed(const QString &errorMessage);

  // Connection status signals
  void connectionCheckStarted();
  void connectionCheckSucceeded();
  void connectionCheckFailed(const QString &errorMessage);

private slots:
  void handleTimeout();
  void handleStreamData();

private:
  QNetworkAccessManager *networkManager();
  QUrl modelsEndpoint() const;

  void sendChatCompletionInternal(const QString &systemPrompt,
                                   const QString &userPrompt,
                                   double temperature,
                                   bool streaming,
                                   int retryCount = 0);
  void setupRequestTimeout(QNetworkReply *reply);
  bool shouldRetry(QNetworkReply::NetworkError error) const;
  int calculateBackoffDelay(int retryCount) const;

  // Configuration
  QUrl m_endpoint;
  QString m_model;
  QString m_apiKey;
  int m_timeoutMs;
  int m_maxRetries;
  int m_retryDelayMs;

  // Runtime state
  QNetworkAccessManager *m_networkManager;
  QNetworkReply *m_currentReply;
  QTimer *m_timeoutTimer;
  bool m_requestTimedOut;

  // Retry tracking
  QString m_lastSystemPrompt;
  QString m_lastUserPrompt;
  double m_lastTemperature;
  bool m_lastStreaming;
  int m_currentRetryCount;
};

#endif // LMSTUDIO_CLIENT_H
