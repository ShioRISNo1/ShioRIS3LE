#ifndef COLLAPSIBLE_GROUP_BOX_H
#define COLLAPSIBLE_GROUP_BOX_H

#include <QHBoxLayout>
#include <QToolButton>
#include <QVBoxLayout>
#include <QWidget>
#include <QColor>

class CollapsibleGroupBox : public QWidget {
  Q_OBJECT
public:
  explicit CollapsibleGroupBox(const QString &title, QWidget *parent = nullptr);

  void setContentLayout(QLayout *layout);
  void setTitle(const QString &title);
  void addHeaderWidget(QWidget *widget, bool insertBeforeToggle = false);
  void setCollapsed(bool collapsed);
  void setTitleColor(const QColor &color = QColor());

signals:
  void toggled(bool expanded);

private slots:
  void toggle(bool checked);

private:
  QToolButton *m_toggleButton;
  QWidget *m_contentWidget;
  QHBoxLayout *m_headerLayout;
};

#endif // COLLAPSIBLE_GROUP_BOX_H
