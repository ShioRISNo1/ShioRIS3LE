#include "visualization/collapsible_group_box.h"

CollapsibleGroupBox::CollapsibleGroupBox(const QString &title, QWidget *parent)
    : QWidget(parent) {
  m_toggleButton = new QToolButton(this);
  m_toggleButton->setStyleSheet("QToolButton { border: none; }");
  m_toggleButton->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
  m_toggleButton->setArrowType(Qt::DownArrow);
  m_toggleButton->setText(title);
  m_toggleButton->setCheckable(true);
  m_toggleButton->setChecked(true);
  connect(m_toggleButton, &QToolButton::toggled, this,
          &CollapsibleGroupBox::toggle);

  m_contentWidget = new QWidget(this);

  QWidget *headerWidget = new QWidget(this);
  m_headerLayout = new QHBoxLayout(headerWidget);
  m_headerLayout->setContentsMargins(0, 0, 0, 0);
  m_headerLayout->addWidget(m_toggleButton);
  m_headerLayout->addStretch();

  QVBoxLayout *layout = new QVBoxLayout(this);
  layout->setSpacing(0);
  layout->setContentsMargins(0, 0, 0, 0);
  layout->addWidget(headerWidget);
  layout->addWidget(m_contentWidget);
}

void CollapsibleGroupBox::setContentLayout(QLayout *layout) {
  m_contentWidget->setLayout(layout);
}

void CollapsibleGroupBox::setTitle(const QString &title) {
  m_toggleButton->setText(title);
}

void CollapsibleGroupBox::addHeaderWidget(QWidget *widget,
                                          bool insertBeforeToggle) {
  if (insertBeforeToggle) {
    m_headerLayout->insertWidget(0, widget);
  } else {
    // Insert before the stretch so widgets stay aligned to the left
    m_headerLayout->insertWidget(m_headerLayout->count() - 1, widget);
  }
}

void CollapsibleGroupBox::toggle(bool checked) {
  m_contentWidget->setVisible(checked);
  m_toggleButton->setArrowType(checked ? Qt::DownArrow : Qt::RightArrow);
  emit toggled(checked);
}

void CollapsibleGroupBox::setCollapsed(bool collapsed) {
  // checked=true means expanded
  bool wantChecked = !collapsed;
  if (m_toggleButton->isChecked() != wantChecked) {
    m_toggleButton->setChecked(wantChecked); // will trigger toggle()
  } else {
    // Ensure state is consistent even if already in desired state
    toggle(wantChecked);
  }
}

void CollapsibleGroupBox::setTitleColor(const QColor &color) {
  if (color.isValid()) {
    m_toggleButton->setStyleSheet(
        QString("QToolButton { border: none; color: %1; }").arg(color.name()));
  } else {
    m_toggleButton->setStyleSheet("QToolButton { border: none; }");
  }
}
