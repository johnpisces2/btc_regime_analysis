import json
import re
import sys
from pathlib import Path

from PySide6.QtCore import QDate, QProcess, QTimer, Qt
from PySide6.QtGui import QPixmap, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


BASE_DIR = Path(__file__).resolve().parent
RESULT_FILES = {
    "Market Classification": BASE_DIR / "results" / "market_classification.png",
    "State Summary": BASE_DIR / "results" / "state_summary.png",
    "Regime Distribution": BASE_DIR / "results" / "regime_distribution.png",
}

REGIME_TEXT = {
    "Consolidation": "Consolidation",
    "Bull": "Bull",
    "Bear": "Bear",
    "Consolidation (盤整)": "Consolidation",
    "Bull (牛市)": "Bull",
    "Bear (熊市)": "Bear",
}


class BadgeLabel(QLabel):
    STYLES = {
        "neutral": ("#F6EEE2", "#7A5A35", "#D8C3A6"),
        "info": ("#E5F3F1", "#0F766E", "#8BC8C0"),
        "success": ("#E7F7EC", "#1F7A45", "#96D0A8"),
        "warning": ("#FFF2E2", "#B45309", "#E8BE87"),
        "danger": ("#FCE9E7", "#B54745", "#E1A19B"),
    }

    def __init__(self, text="", variant="neutral"):
        super().__init__(text)
        self.setAlignment(Qt.AlignCenter)
        self.set_variant(variant)

    def set_variant(self, variant):
        background, foreground, border = self.STYLES.get(variant, self.STYLES["neutral"])
        self.setStyleSheet(
            f"""
            QLabel {{
                background: {background};
                color: {foreground};
                border: 1px solid {border};
                border-radius: 999px;
                padding: 4px 10px;
                font-size: 11px;
                font-weight: 700;
            }}
            """
        )


class MetricCard(QFrame):
    ACCENTS = {
        "gold": ("#FFF5E8", "#C56A27", "#E8C39B"),
        "teal": ("#E9F8F5", "#0F766E", "#91D2C9"),
        "slate": ("#EFF3F7", "#334155", "#C4D0DD"),
        "coral": ("#FFF0EA", "#B85A33", "#E4B29F"),
    }

    def __init__(self, title, accent):
        super().__init__()
        background, value_color, border = self.ACCENTS.get(accent, self.ACCENTS["slate"])
        self.setStyleSheet(
            f"""
            QFrame {{
                background: {background};
                border: 1px solid {border};
                border-radius: 18px;
            }}
            QLabel {{
                background: transparent;
                border: none;
            }}
            QLabel#metricTitle {{
                color: #6B7280;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }}
            QLabel#metricValue {{
                color: {value_color};
                font-size: 24px;
                font-weight: 800;
            }}
            QLabel#metricNote {{
                color: #475467;
                font-size: 11px;
            }}
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 13, 15, 13)
        layout.setSpacing(6)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("metricTitle")
        layout.addWidget(self.title_label)

        self.value_label = QLabel("--")
        self.value_label.setObjectName("metricValue")
        layout.addWidget(self.value_label)

        self.note_label = QLabel("Awaiting fresh data.")
        self.note_label.setObjectName("metricNote")
        self.note_label.setWordWrap(False)
        self.note_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(self.note_label)

    def set_data(self, value, note):
        self.value_label.setText(value)
        self.note_label.setText(note)


class ImagePreview(QLabel):
    def __init__(self, image_path):
        super().__init__()
        self.image_path = Path(image_path)
        self._pixmap = None
        self.setObjectName("imagePreview")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setWordWrap(True)
        self._set_placeholder("No chart loaded yet", "Run All to generate fresh visuals.")

    def _set_placeholder(self, title, subtitle):
        self._pixmap = None
        self.setPixmap(QPixmap())
        self.setText(
            f"""
            <div style="text-align:center;">
              <div style="font-size:22px; font-weight:800; color:#7A5A35; margin-bottom:8px;">{title}</div>
              <div style="font-size:13px; color:#6B7280;">{subtitle}</div>
            </div>
            """
        )

    def clear_preview(self):
        self._set_placeholder("No chart loaded yet", "Run All to generate fresh visuals.")

    def refresh(self):
        if not self.image_path.exists():
            self._set_placeholder("Chart missing", self.image_path.name)
            return

        pixmap = QPixmap(str(self.image_path))
        if pixmap.isNull():
            self._set_placeholder("Unable to load chart", self.image_path.name)
            return

        self._pixmap = pixmap
        self.setText("")
        self.setPixmap(QPixmap())

    def render_for_width(self, target_width):
        if self._pixmap is None:
            return
        width = max(320, min(int(target_width), self._pixmap.width()))
        scaled = self._pixmap.scaledToWidth(width, Qt.SmoothTransformation)
        self.setPixmap(scaled)
        self.resize(scaled.size())


class ChartScrollArea(QScrollArea):
    def __init__(self, preview):
        super().__init__()
        self.preview = preview
        self.setWidgetResizable(False)
        self.setFrameShape(QScrollArea.NoFrame)
        self.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setWidget(preview)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.sync_preview()

    def showEvent(self, event):
        super().showEvent(event)
        self.sync_preview()

    def sync_preview(self):
        viewport_width = self.viewport().width() - 12
        self.preview.render_for_width(viewport_width)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BTC Regime Analysis")
        self.resize(1700, 860)
        self.setMinimumWidth(1600)
        self.setMinimumHeight(860)
        self.setMaximumHeight(2160)

        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(BASE_DIR))
        self.process.readyReadStandardOutput.connect(self.read_stdout)
        self.process.readyReadStandardError.connect(self.read_stderr)
        self.process.finished.connect(self.on_process_finished)
        self.process.errorOccurred.connect(self.on_process_error)

        self.command_queue = []
        self.current_command = None
        self.latest_status_text = "No results loaded yet. Click Run All to generate fresh outputs."
        self.latest_probabilities = {}
        self.latest_regime = None
        self.latest_state = None
        self.latest_price = None
        self.latest_window_share = {}
        self.latest_avg_prob = {}

        self.image_widgets = {}
        self.chart_scroll_areas = {}
        self._build_ui()
        self._apply_styles()
        self.reset_views()
        self._refresh_context_badges()
        self._set_pipeline_status("Ready to run", "neutral")
        self.clear_initial_focus()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(14, 12, 14, 14)
        main_layout.setSpacing(10)

        header = self._build_header()
        main_layout.addWidget(header)

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        self.main_splitter.addWidget(self._build_left_panel())
        self.main_splitter.addWidget(self._build_right_panel())
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([480, 1320])
        main_layout.addWidget(self.main_splitter)

        self.statusBar().showMessage("Ready")
        QTimer.singleShot(0, self._apply_layout_sizes)

    def _apply_styles(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #F6F1E8;
                color: #1F2933;
                font-family: "Avenir Next", "Segoe UI", "PingFang TC", "Noto Sans TC";
                font-size: 14px;
            }
            QFrame#heroCard, QFrame#sidePanel, QFrame#heroPulsePanel {
                background: #FFFDF8;
                border: 1px solid #D9CEBF;
                border-radius: 24px;
            }
            QFrame#insightPanel, QFrame#tonePanel {
                background: #F7F1E7;
                border: 1px solid #E1D4C2;
                border-radius: 18px;
            }
            QLabel {
                background: transparent;
                color: #1F2933;
            }
            QLabel#eyebrow {
                color: #B85A33;
                font-size: 11px;
                font-weight: 800;
                letter-spacing: 0.12em;
                text-transform: uppercase;
            }
            QLabel#heroTitle {
                color: #1D2939;
                font-size: 28px;
                font-weight: 800;
            }
            QLabel#heroSubtitle {
                color: #475467;
                font-size: 13px;
            }
            QLabel#sectionHint {
                color: #667085;
                font-size: 12px;
            }
            QLabel#fieldLabel {
                color: #344054;
                font-size: 13px;
                font-weight: 700;
                min-width: 110px;
            }
            QLabel#subheading {
                color: #344054;
                font-size: 13px;
                font-weight: 700;
            }
            QLabel#narrativeText {
                color: #344054;
                font-size: 14px;
                line-height: 1.45;
            }
            QLabel#detailText {
                color: #475467;
                font-size: 12px;
                line-height: 1.45;
            }
            QLabel#imagePreview {
                background: #FBF8F2;
                border: 1px dashed #D8C3A6;
                border-radius: 18px;
                padding: 12px;
            }
            QGroupBox {
                background: #FFFDF8;
                border: 1px solid #D9CEBF;
                border-radius: 18px;
                margin-top: 14px;
                padding-top: 16px;
                font-weight: 800;
                color: #1D2939;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                top: 12px;
                padding: 0 4px;
                color: #344054;
                background: transparent;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QDateEdit {
                background: #FFF4E6;
                color: #18212B;
                border: 1px solid #CDAE81;
                border-radius: 10px;
                padding: 6px 8px;
                min-height: 18px;
                selection-background-color: #0F766E;
                selection-color: #FFFFFF;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QDateEdit:focus {
                border: 1px solid #0F766E;
                background: #FFF9F1;
            }
            QComboBox::drop-down, QSpinBox::down-button, QSpinBox::up-button,
            QDoubleSpinBox::down-button, QDoubleSpinBox::up-button, QDateEdit::drop-down {
                border: none;
                width: 24px;
                background: transparent;
            }
            QCheckBox {
                spacing: 8px;
                color: #24303D;
                font-weight: 700;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 5px;
                border: 1px solid #BFA98A;
                background: #FFF8EF;
            }
            QCheckBox::indicator:checked {
                background: #0F766E;
                border: 1px solid #0F766E;
            }
            QPushButton {
                border-radius: 12px;
                padding: 8px 14px;
                border: 1px solid #D5C2A5;
                background: #FFF8EF;
                color: #6A421F;
                font-weight: 800;
                min-height: 18px;
            }
            QPushButton:hover {
                background: #FFF2DE;
            }
            QPushButton:disabled {
                background: #EFE6D9;
                color: #98A2B3;
                border-color: #D7C9B7;
            }
            QPushButton#primaryButton {
                background: #B85A33;
                color: #FFFFFF;
                border: 1px solid #B85A33;
            }
            QPushButton#primaryButton:hover {
                background: #A24D29;
            }
            QPushButton#secondaryButton {
                background: #FFFDF8;
                color: #6B4A23;
                border: 1px solid #D5C2A5;
            }
            QTabWidget::pane {
                border: 1px solid #DACFBE;
                border-radius: 16px;
                background: #FFFDF8;
                margin-top: 8px;
            }
            QTabWidget::tab-bar {
                alignment: center;
            }
            QTabBar::tab {
                text-align: center;
            }
            QTabWidget#contentTabs::pane {
                border: none;
                background: transparent;
                margin-top: 4px;
            }
            QTabWidget#contentTabs {
                background: transparent;
            }
            QTabWidget#contentTabs QStackedWidget {
                background: transparent;
            }
            QTabWidget#contentTabs QTabBar {
                background: transparent;
            }
            QTabWidget#contentTabs::tab-bar {
                alignment: center;
            }
            QTabBar::tab {
                background: #EDE3D3;
                color: #7A5A35;
                border: none;
                padding: 7px 12px;
                margin-right: 4px;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                min-width: 110px;
                font-weight: 800;
            }
            QTabBar::tab:selected {
                background: #0F766E;
                color: #FFFFFF;
            }
            QTabBar::tab:!selected:hover {
                background: #E4D7C3;
            }
            QTabWidget#contentTabs QTabBar::tab {
                min-width: 150px;
                padding: 9px 16px;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
            QSplitter::handle {
                background: #D9CEBF;
                width: 8px;
                height: 8px;
                border-radius: 4px;
            }
            QStatusBar {
                background: #EFE6DA;
                color: #6B7280;
            }
            QTextEdit, QPlainTextEdit {
                background: #FBF8F2;
                color: #24303D;
                border: 1px solid #D9CEBF;
                border-radius: 14px;
                padding: 8px;
            }
            QPlainTextEdit {
                font-family: "SF Mono", "Menlo", "Consolas", monospace;
            }
            QScrollBar:vertical {
                background: #E5D9C8;
                width: 12px;
                margin: 4px 2px 4px 2px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #9E8B73;
                min-height: 28px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background: #7A6B55;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
                border: none;
                background: transparent;
            }
            QScrollBar:horizontal {
                background: #E5D9C8;
                height: 12px;
                margin: 2px 4px 2px 4px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal {
                background: #9E8B73;
                min-width: 28px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #7A6B55;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
                border: none;
                background: transparent;
            }
            QScrollBar::handle:vertical {
                background: #C7B59C;
                min-height: 28px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background: #B79E7D;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
                border: none;
                background: transparent;
            }
            QScrollBar:horizontal {
                background: #F0E8DB;
                height: 12px;
                margin: 2px 4px 2px 4px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal {
                background: #C7B59C;
                min-width: 28px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #B79E7D;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
                border: none;
                background: transparent;
            }
            """
        )

    def clear_initial_focus(self):
        self.symbol_input.deselect()
        self.symbol_input.clearFocus()
        self.centralWidget().setFocusPolicy(Qt.StrongFocus)
        self.centralWidget().setFocus()

    def _build_header(self):
        header = QFrame()
        header.setObjectName("heroCard")
        header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        root_layout = QVBoxLayout(header)
        root_layout.setContentsMargins(18, 16, 18, 16)
        root_layout.setSpacing(12)

        layout = QHBoxLayout()
        layout.setSpacing(14)

        left_col = QVBoxLayout()
        left_col.setSpacing(6)

        eyebrow = QLabel("BTC REGIME ANALYSIS")
        eyebrow.setObjectName("eyebrow")
        left_col.addWidget(eyebrow)

        title = QLabel("BTC Regime分析")
        title.setObjectName("heroTitle")
        left_col.addWidget(title)

        subtitle = QLabel(
            "從歷史資料到模型訓練，用單一工作區追蹤BTC市場Regime與後驗概率。"
        )
        subtitle.setObjectName("heroSubtitle")
        subtitle.setWordWrap(True)
        left_col.addWidget(subtitle)

        badge_row = QHBoxLayout()
        badge_row.setSpacing(6)
        self.symbol_badge = BadgeLabel()
        self.model_badge = BadgeLabel(variant="info")
        self.window_badge = BadgeLabel()
        badge_row.addWidget(self.symbol_badge)
        badge_row.addWidget(self.model_badge)
        badge_row.addWidget(self.window_badge)
        badge_row.addStretch(1)
        left_col.addLayout(badge_row)

        layout.addLayout(left_col, 1)

        right_col = QVBoxLayout()
        right_col.setSpacing(8)
        right_col.setAlignment(Qt.AlignTop | Qt.AlignRight)

        status_hint = QLabel("Pipeline Status")
        status_hint.setObjectName("sectionHint")
        status_hint.setAlignment(Qt.AlignRight)
        right_col.addWidget(status_hint)

        self.pipeline_badge = BadgeLabel("Ready to run", "neutral")
        right_col.addWidget(self.pipeline_badge, alignment=Qt.AlignRight)

        button_row = QHBoxLayout()
        button_row.setSpacing(8)

        self.clear_button = QPushButton("Reset View")
        self.clear_button.setObjectName("secondaryButton")
        self.clear_button.clicked.connect(self.clear_results)
        button_row.addWidget(self.clear_button)

        self.run_all_button = QPushButton("Run All")
        self.run_all_button.setObjectName("primaryButton")
        self.run_all_button.clicked.connect(self.run_all)
        button_row.addWidget(self.run_all_button)

        right_col.addLayout(button_row)
        right_col.addStretch(1)
        layout.addLayout(right_col)
        root_layout.addLayout(layout)
        root_layout.addWidget(self._build_header_pulse_panel())

        return header

    def _build_header_pulse_panel(self):
        panel = QFrame()
        panel.setObjectName("heroPulsePanel")

        layout = QHBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        metrics_col = QVBoxLayout()
        metrics_col.setSpacing(6)

        metrics = QGridLayout()
        metrics.setHorizontalSpacing(8)
        metrics.setVerticalSpacing(6)

        self.regime_card = MetricCard("Market Regime", "gold")
        self.price_card = MetricCard("BTC Price", "teal")
        self.state_card = MetricCard("Latent State", "slate")
        self.confidence_card = MetricCard("Top Confidence", "coral")

        metrics.addWidget(self.regime_card, 0, 0)
        metrics.addWidget(self.price_card, 0, 1)
        metrics.addWidget(self.state_card, 0, 2)
        metrics.addWidget(self.confidence_card, 0, 3)
        metrics_col.addLayout(metrics)

        self.summary_label = QLabel("Execute the pipeline to generate the latest regime assessment.")
        self.summary_label.setObjectName("narrativeText")
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.summary_label.setMinimumHeight(36)
        metrics_col.addWidget(self.summary_label)

        layout.addLayout(metrics_col, 3)

        detail_col = QVBoxLayout()
        detail_col.setSpacing(6)

        self.probability_label = QLabel("Posterior: --")
        self.probability_label.setObjectName("detailText")
        self.probability_label.setWordWrap(True)
        self.probability_label.setTextFormat(Qt.RichText)
        detail_col.addWidget(self.probability_label)

        self.market_texture_label = QLabel("Window: --")
        self.market_texture_label.setObjectName("detailText")
        self.market_texture_label.setWordWrap(True)
        self.market_texture_label.setTextFormat(Qt.RichText)
        detail_col.addWidget(self.market_texture_label)

        layout.addLayout(detail_col, 1)
        return panel

    def _build_left_panel(self):
        panel_scroll = QScrollArea()
        panel_scroll.setWidgetResizable(True)
        panel_scroll.setFrameShape(QScrollArea.NoFrame)
        panel_scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        panel_scroll.setMinimumWidth(400)

        panel = QFrame()
        panel.setObjectName("sidePanel")
        panel.setMinimumWidth(440)
        panel.setMaximumWidth(460)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(self._build_data_group())
        layout.addWidget(self._build_train_group())
        layout.addWidget(self._build_playbook_group())
        layout.addStretch(1)

        panel_scroll.setWidget(panel)
        return panel_scroll

    def _build_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.content_tabs = QTabWidget()
        self.content_tabs.setObjectName("contentTabs")
        self.content_tabs.addTab(self._build_chart_panel(), "Visual Story")
        self.content_tabs.addTab(self._build_workspace_tabs(), "Model Notes")
        self.content_tabs.tabBar().setExpanding(True)
        layout.addWidget(self.content_tabs, 1)
        return panel

    def _apply_layout_sizes(self):
        if hasattr(self, "main_splitter"):
            total_width = max(self.main_splitter.width(), 1660)
            left_width = 460
            self.main_splitter.setSizes([left_width, max(total_width - left_width, 1100)])

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self._apply_layout_sizes)

    def _build_chart_panel(self):
        panel = QWidget()
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        tabs = QTabWidget()
        tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        tabs.currentChanged.connect(lambda _index: self._sync_chart_previews())
        for title, path in RESULT_FILES.items():
            preview = ImagePreview(path)
            scroll = ChartScrollArea(preview)
            tabs.addTab(scroll, title)
            self.image_widgets[title] = preview
            self.chart_scroll_areas[title] = scroll
        tabs.tabBar().setExpanding(True)
        self.chart_tabs = tabs
        layout.addWidget(tabs, 1)
        return panel

    def _build_workspace_tabs(self):
        tabs = QTabWidget()
        tabs.addTab(self._wrap_in_scroll(self._build_metadata_group()), "Model Diagnostics")
        tabs.addTab(self._wrap_in_scroll(self._build_log_group()), "Execution Log")
        tabs.tabBar().setExpanding(True)
        return tabs

    def _wrap_in_scroll(self, widget):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setAlignment(Qt.AlignTop)
        scroll.setWidget(widget)
        return scroll

    def _make_field_label(self, text):
        label = QLabel(text)
        label.setObjectName("fieldLabel")
        label.setMinimumWidth(110)
        return label

    def _build_data_group(self):
        group = QGroupBox("Data Configuration")
        layout = QFormLayout(group)
        layout.setContentsMargins(12, 14, 12, 12)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(8)
        layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        intro = QLabel("設定資料來源、時間區間與圖表視窗。")
        intro.setObjectName("sectionHint")
        intro.setWordWrap(True)
        layout.addRow(intro)

        self.symbol_input = QLineEdit("BTC/USDT")
        self.symbol_input.setMinimumWidth(170)
        self.symbol_input.textChanged.connect(self._refresh_context_badges)
        layout.addRow(self._make_field_label("Symbol"), self.symbol_input)

        self.exchange_input = QComboBox()
        self.exchange_input.setMinimumWidth(170)
        self.exchange_input.addItems(["binance", "okx", "bybit", "kraken", "bitget"])
        self.exchange_input.currentTextChanged.connect(self._refresh_context_badges)
        layout.addRow(self._make_field_label("Exchange"), self.exchange_input)

        self.timeframe_input = QComboBox()
        self.timeframe_input.setMinimumWidth(170)
        self.timeframe_input.addItems(["1d", "4h", "1h"])
        self.timeframe_input.setCurrentText("4h")
        self.timeframe_input.currentTextChanged.connect(self._refresh_context_badges)
        layout.addRow(self._make_field_label("Timeframe"), self.timeframe_input)

        self.start_date_input = QDateEdit()
        self.start_date_input.setMinimumWidth(170)
        self.start_date_input.setCalendarPopup(True)
        self.start_date_input.setDate(QDate(2018, 1, 1))
        self.start_date_input.setDisplayFormat("yyyy-MM-dd")
        layout.addRow(self._make_field_label("Start Date"), self.start_date_input)

        end_row = QWidget()
        end_row.setStyleSheet("background: transparent;")
        end_layout = QHBoxLayout(end_row)
        end_layout.setContentsMargins(0, 0, 0, 0)
        end_layout.setSpacing(6)

        self.end_date_input = QDateEdit()
        self.end_date_input.setMinimumWidth(150)
        self.end_date_input.setCalendarPopup(True)
        self.end_date_input.setDate(QDate.currentDate())
        self.end_date_input.setDisplayFormat("yyyy-MM-dd")
        end_layout.addWidget(self.end_date_input)

        self.use_today_checkbox = QCheckBox("Current Date")
        self.use_today_checkbox.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.use_today_checkbox.setChecked(True)
        self.use_today_checkbox.toggled.connect(self.end_date_input.setDisabled)
        self.end_date_input.setDisabled(True)
        end_layout.addWidget(self.use_today_checkbox)
        end_layout.addStretch(1)
        layout.addRow(self._make_field_label("End Date"), end_row)

        self.display_days_input = QSpinBox()
        self.display_days_input.setRange(30, 9999999)
        self.display_days_input.setValue(365)
        self.display_days_input.setMinimumWidth(120)
        self.display_days_input.valueChanged.connect(self._refresh_context_badges)
        layout.addRow(self._make_field_label("Chart Window"), self.display_days_input)

        return group

    def _build_train_group(self):
        group = QGroupBox("Model Configuration")
        layout = QGridLayout(group)
        layout.setContentsMargins(12, 14, 12, 12)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(8)

        hint = QLabel("設定模型類型、state 數量、標準化方式與測試區間。")
        hint.setObjectName("sectionHint")
        hint.setWordWrap(True)
        layout.addWidget(hint, 0, 0, 1, 4)

        self.model_input = QComboBox()
        self.model_input.addItems(["hmm", "kmeans"])
        self.model_input.currentTextChanged.connect(self.update_model_controls)
        layout.addWidget(self._make_field_label("Model"), 1, 0)
        layout.addWidget(self.model_input, 1, 1)

        self.states_input = QSpinBox()
        self.states_input.setRange(2, 10)
        self.states_input.setValue(3)
        self.states_input.valueChanged.connect(self._refresh_context_badges)
        layout.addWidget(self._make_field_label("States"), 1, 2)
        layout.addWidget(self.states_input, 1, 3)

        self.scaler_input = QComboBox()
        self.scaler_input.addItems(["standard", "minmax"])
        layout.addWidget(self._make_field_label("Scaler"), 2, 0)
        layout.addWidget(self.scaler_input, 2, 1)

        self.test_size_input = QDoubleSpinBox()
        self.test_size_input.setRange(0.05, 0.5)
        self.test_size_input.setSingleStep(0.05)
        self.test_size_input.setDecimals(2)
        self.test_size_input.setValue(0.20)
        layout.addWidget(self._make_field_label("Test Size"), 2, 2)
        layout.addWidget(self.test_size_input, 2, 3)

        self.min_regime_run_input = QSpinBox()
        self.min_regime_run_input.setRange(1, 48)
        self.min_regime_run_input.setValue(6)
        self.min_regime_run_label = self._make_field_label("Min Regime Run")
        layout.addWidget(self.min_regime_run_label, 3, 0)
        layout.addWidget(self.min_regime_run_input, 3, 1)

        self.update_model_controls(self.model_input.currentText())
        return group

    def _build_playbook_group(self):
        group = QGroupBox("Runbook")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(12, 14, 12, 12)
        layout.setSpacing(8)

        title = QLabel("Recommended Workflow")
        title.setObjectName("subheading")
        layout.addWidget(title)

        notes = QLabel(
            "1. 以預設設定執行 `hmm` 建立基準結果。\n"
            "2. 優先檢視 `Market Pulse` 的 regime、confidence 與 recent window distribution。\n"
            "3. 若需比較聚類結果，再切換至 `kmeans` 並調整 `Min Regime Run`。"
        )
        notes.setObjectName("detailText")
        notes.setWordWrap(True)
        layout.addWidget(notes)

        tone_panel = QFrame()
        tone_panel.setObjectName("tonePanel")
        tone_layout = QVBoxLayout(tone_panel)
        tone_layout.setContentsMargins(12, 10, 12, 10)
        tone_layout.setSpacing(5)

        tone_title = QLabel("Interface Focus")
        tone_title.setObjectName("subheading")
        tone_layout.addWidget(tone_title)

        tone_body = QLabel(
            "優先呈現 regime snapshot 與主要圖表；模型細節與執行紀錄則收納於次層分頁。"
        )
        tone_body.setObjectName("detailText")
        tone_body.setWordWrap(True)
        tone_layout.addWidget(tone_body)

        layout.addWidget(tone_panel)
        return group

    def _build_metadata_group(self):
        group = QGroupBox("Model Diagnostics")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(12, 14, 12, 12)
        layout.setSpacing(8)

        hint = QLabel("彙整訓練後 metrics、transition matrix 與各 state 的摘要統計。")
        hint.setObjectName("sectionHint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.metadata_text = QTextEdit()
        self.metadata_text.setReadOnly(True)
        self.metadata_text.setMinimumHeight(180)
        layout.addWidget(self.metadata_text)
        return group

    def _build_log_group(self):
        group = QGroupBox("Execution Log")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(12, 14, 12, 12)
        layout.setSpacing(8)

        highlights_label = QLabel("Run Highlights")
        highlights_label.setObjectName("subheading")
        layout.addWidget(highlights_label)

        self.log_highlights = QTextEdit()
        self.log_highlights.setReadOnly(True)
        self.log_highlights.setMinimumHeight(120)
        layout.addWidget(self.log_highlights)

        raw_label = QLabel("Raw Logs")
        raw_label.setObjectName("subheading")
        layout.addWidget(raw_label)

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output)
        return group

    def _set_pipeline_status(self, text, variant):
        self.pipeline_badge.setText(text)
        self.pipeline_badge.set_variant(variant)

    def _refresh_context_badges(self, *_):
        if not hasattr(self, "symbol_badge"):
            return
        symbol = self.symbol_input.text().strip() or "BTC/USDT"
        exchange = self.exchange_input.currentText()
        timeframe = self.timeframe_input.currentText()
        model = self.model_input.currentText().upper()
        states = self.states_input.value()
        days = self.display_days_input.value()

        self.symbol_badge.setText(f"{symbol} · {exchange}")
        self.model_badge.setText(f"{model} · {states} states")
        self.window_badge.setText(f"{days}d window · {timeframe}")

    def append_log(self, text):
        if not text:
            return
        self.log_output.moveCursor(QTextCursor.End)
        self.log_output.insertPlainText(text)
        self.log_output.moveCursor(QTextCursor.End)
        self.log_output.ensureCursorVisible()
        self._parse_status_from_output(text)
        self._refresh_log_highlights()

    def _parse_status_from_output(self, text):
        price_match = re.search(r"(?:BTC price|比特幣價格):\s*\$([0-9,]+\.\d+)", text)
        regime_match = re.search(r"(?:Market regime|市場分類):\s*(.+)", text)
        state_match = re.search(r"(?:Latent state|潛在狀態):\s*(state \d+)", text)
        consolidation_match = re.search(r"(?:Consolidation|盤整):\s*([0-9.]+)%", text)
        bull_match = re.search(r"(?:Bull|牛市):\s*([0-9.]+)%", text)
        bear_match = re.search(r"(?:Bear|熊市):\s*([0-9.]+)%", text)

        parts = []
        regime_label = None
        if regime_match:
            regime_label = regime_match.group(1).strip()
            self.latest_regime = REGIME_TEXT.get(regime_label, regime_label)
            parts.append(regime_label)
        if state_match:
            self.latest_state = state_match.group(1).strip()
            parts.append(self.latest_state)
        if price_match:
            self.latest_price = price_match.group(1)
            parts.append(f"${self.latest_price}")

        probabilities = {}
        if consolidation_match:
            probabilities["Consolidation"] = float(consolidation_match.group(1))
        if bull_match:
            probabilities["Bull"] = float(bull_match.group(1))
        if bear_match:
            probabilities["Bear"] = float(bear_match.group(1))
        if probabilities:
            self.latest_probabilities = probabilities

        if parts:
            self.latest_status_text = " | ".join(parts)
        self._render_summary()

    def _build_human_summary(self, regime_label, probabilities):
        if not probabilities:
            if regime_label:
                return f"The latest classification indicates {REGIME_TEXT.get(regime_label, regime_label)}."
            return ""

        ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        primary_name, primary_score = ordered[0]

        if len(ordered) == 1:
            return f"The posterior distribution is fully concentrated in {primary_name} ({primary_score:.1f}%)."

        secondary_name, secondary_score = ordered[1]
        gap = primary_score - secondary_score

        if primary_score >= 85:
            return f"The posterior distribution indicates a strong {primary_name} regime with {primary_score:.1f}% confidence."
        if gap >= 20:
            return (
                f"The model favors {primary_name} at {primary_score:.1f}%, "
                f"while {secondary_name} remains the secondary scenario at {secondary_score:.1f}%."
            )
        return (
            f"The current read is balanced: {primary_name} leads marginally over {secondary_name} "
            f"({primary_score:.1f}% vs {secondary_score:.1f}%)."
        )

    def _format_probability_html(self, probabilities):
        if not probabilities:
            return "<span style='color:#667085;'>No posterior probabilities available.</span>"

        chip_styles = {
            "Consolidation": ("#EEF2F6", "#475467", "#CBD5E1"),
            "Bull": ("#E7F7EC", "#1F7A45", "#96D0A8"),
            "Bear": ("#FCE9E7", "#B54745", "#E1A19B"),
        }
        ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        chips = []
        for name, score in ordered:
            background, foreground, border = chip_styles.get(name, chip_styles["Consolidation"])
            chips.append(
                f"<span style='display:inline-block; margin: 4px 10px 0 0; padding: 6px 12px; "
                f"border-radius: 999px; background:{background}; color:{foreground}; "
                f"border:1px solid {border};'><b>{name}</b> {score:.1f}%</span>"
            )
        return "".join(chips)

    def _format_texture_html(self):
        sections = []
        if self.latest_window_share:
            sections.append(("Recent window distribution", self.latest_window_share))
        if self.latest_avg_prob:
            sections.append(("Average posterior", self.latest_avg_prob))

        if not sections:
            return "<span style='color:#667085;'>Recent window distribution and average posterior will be populated after execution.</span>"

        blocks = []
        for title, values in sections:
            ordered = " · ".join(
                f"<b>{name}</b> {values.get(name, 0):.1f}%"
                for name in ("Consolidation", "Bull", "Bear")
            )
            blocks.append(f"<div style='margin-bottom:4px;'><span style='color:#344054; font-weight:700;'>{title}:</span> {ordered}</div>")
        return "".join(blocks)

    def _format_probability_compact(self, probabilities):
        if not probabilities:
            return "<span style='color:#667085;'>Posterior: No data</span>"
        ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        parts = []
        for name, score in ordered:
            parts.append(f"<b>{name}</b> {score:.1f}%")
        return f"Posterior: {' | '.join(parts)}"

    def _format_texture_compact(self):
        if not self.latest_window_share:
            return "<span style='color:#667085;'>Window: No data</span>"
        ordered = " | ".join(
            f"<b>{name}</b> {self.latest_window_share.get(name, 0):.1f}%"
            for name in ("Consolidation", "Bull", "Bear")
        )
        return f"Window: {ordered}"

    def _render_summary(self):
        if not hasattr(self, "regime_card"):
            return
        regime = self.latest_regime or "Awaiting run"
        state = self.latest_state or "N/A"
        price = f"${self.latest_price}" if self.latest_price else "--"
        model_name = self.model_input.currentText().upper()
        symbol = self.symbol_input.text().strip() or "BTC/USDT"
        exchange = self.exchange_input.currentText()
        timeframe = self.timeframe_input.currentText()

        narrative = (
            self._build_human_summary(self.latest_regime, self.latest_probabilities)
            if self.latest_regime
            else "Execute `Run All` to refresh data, retrain the selected model, and generate the current regime assessment."
        )

        if self.latest_probabilities:
            ordered = sorted(self.latest_probabilities.items(), key=lambda item: item[1], reverse=True)
            top_name, top_score = ordered[0]
            confidence_value = f"{top_score:.1f}%"
            confidence_note = f"{top_name} carries the highest posterior weight."
        else:
            confidence_value = "--"
            confidence_note = "Confidence is available after the next prediction run."

        self.regime_card.set_data(regime, "Latest regime classification generated by the pipeline.")
        self.price_card.set_data(price, f"{symbol} from {exchange} on {timeframe}.")
        self.state_card.set_data(state, f"Latent state reported by {model_name}.")
        self.confidence_card.set_data(confidence_value, confidence_note)

        self.summary_label.setText(narrative)
        self.probability_label.setText(self._format_probability_compact(self.latest_probabilities))
        self.market_texture_label.setText(self._format_texture_compact())

    def _parse_regime_block(self, text, header_pattern):
        pattern = rf"{header_pattern}:\n[-]+\n(.*?)(?:\n\n|$)"
        match = re.search(pattern, text, re.S)
        if not match:
            return {}
        section = match.group(1)
        values = {}
        for name in ("Consolidation", "Bull", "Bear"):
            value_match = re.search(rf"{name}\s*:.*?([0-9.]+)%", section)
            if value_match:
                values[name] = float(value_match.group(1))
        return values

    def _refresh_log_highlights(self):
        text = self.log_output.toPlainText()
        if not text.strip():
            self.log_highlights.setPlainText("No run highlights yet.")
            self._render_summary()
            return

        completed = re.findall(r"Completed:\s*(.+)", text)
        last_completed = completed[-1] if completed else "Running"

        window_match = re.findall(r"Recent (\d+)-day regime summary:", text)
        window_days = window_match[-1] if window_match else str(self.display_days_input.value())

        window_share = self._parse_regime_block(text, rf"Recent {window_days}-day regime summary")
        avg_prob = {}
        avg_prob_match = re.search(r"Average posterior probabilities:\n(.*?)(?:\n\n|$)", text, re.S)
        if avg_prob_match:
            block = avg_prob_match.group(1)
            for name in ("Consolidation", "Bull", "Bear"):
                value_match = re.search(rf"{name}:\s*([0-9.]+)%", block)
                if value_match:
                    avg_prob[name] = float(value_match.group(1))

        if window_share:
            self.latest_window_share = window_share
        if avg_prob:
            self.latest_avg_prob = avg_prob

        highlight_lines = [f"Latest step: {last_completed}"]

        if self.latest_window_share:
            share_line = " | ".join(
                f"{name} {self.latest_window_share.get(name, 0):.1f}%"
                for name in ("Consolidation", "Bull", "Bear")
            )
            highlight_lines.extend(["", f"Window share ({window_days} days)", share_line])

        if self.latest_avg_prob:
            prob_line = " | ".join(
                f"{name} {self.latest_avg_prob.get(name, 0):.1f}%"
                for name in ("Consolidation", "Bull", "Bear")
            )
            highlight_lines.extend(["", "Average posterior", prob_line])

        metric_lines = []
        for label, pattern in (
            ("Train log-likelihood", r"Average log-likelihood per sample \(train\):\s*([\-0-9.]+)"),
            ("Test log-likelihood", r"Average log-likelihood per sample \(test\):\s*([\-0-9.]+)"),
            ("Train silhouette", r"Silhouette score \(train\):\s*([\-0-9.]+)"),
            ("Test silhouette", r"Silhouette score \(test\):\s*([\-0-9.]+)"),
        ):
            matches = re.findall(pattern, text)
            if matches:
                metric_lines.append(f"{label}: {matches[-1]}")

        if metric_lines:
            highlight_lines.extend(["", "Model metrics", *metric_lines])

        if "Command failed" in text or "Process error" in text:
            highlight_lines.extend(["", "Status", "Last run reported an error. Check Raw Logs below."])

        self.log_highlights.setPlainText("\n".join(highlight_lines))
        self._render_summary()

    def build_fetch_args(self):
        args = [
            "collect_data.py",
            "--symbol",
            self.symbol_input.text().strip() or "BTC/USDT",
            "--exchange",
            self.exchange_input.currentText(),
            "--start-date",
            self.start_date_input.date().toString("yyyy-MM-dd"),
            "--timeframe",
            self.timeframe_input.currentText(),
        ]
        if not self.use_today_checkbox.isChecked():
            args.extend(["--end-date", self.end_date_input.date().toString("yyyy-MM-dd")])
        return args

    def build_train_args(self):
        args = [
            "train.py",
            "--model",
            self.model_input.currentText(),
            "--states",
            str(self.states_input.value()),
            "--scaler",
            self.scaler_input.currentText(),
            "--test-size",
            f"{self.test_size_input.value():.2f}",
        ]
        if self.model_input.currentText() == "kmeans":
            args.extend(["--min-regime-run", str(self.min_regime_run_input.value())])
        return args

    def update_model_controls(self, model_name):
        show_min_run = model_name == "kmeans"
        self.min_regime_run_label.setVisible(show_min_run)
        self.min_regime_run_input.setVisible(show_min_run)
        self._refresh_context_badges()
        self._render_summary()

    def build_predict_args(self, update=False):
        args = [
            "predict.py",
            "--days",
            str(self.display_days_input.value()),
        ]
        if update:
            args.extend(
                [
                    "--update",
                    "--symbol",
                    self.symbol_input.text().strip() or "BTC/USDT",
                    "--exchange",
                    self.exchange_input.currentText(),
                    "--timeframe",
                    self.timeframe_input.currentText(),
                ]
            )
        return args

    def run_fetch(self):
        self.start_commands([("Fetch Data", self.build_fetch_args())])

    def run_train(self):
        self.start_commands([("Train Model", self.build_train_args())])

    def run_predict_local(self):
        self.start_commands([("Predict Local", self.build_predict_args(update=False))])

    def run_predict_latest(self):
        self.start_commands([("Predict Latest", self.build_predict_args(update=True))])

    def run_all(self):
        commands = [
            ("Fetch Data", self.build_fetch_args()),
            ("Train Model", self.build_train_args()),
            ("Predict Latest", self.build_predict_args(update=True)),
        ]
        self.start_commands(commands)

    def start_commands(self, commands):
        if self.process.state() != QProcess.NotRunning:
            QMessageBox.information(self, "Busy", "A task is already running.")
            return

        self.command_queue = list(commands)
        self.log_output.clear()
        self.log_highlights.setPlainText("Pipeline started. Highlights will update as each step finishes.")
        self._set_pipeline_status("Pipeline running", "warning")
        self._run_next_command()

    def _run_next_command(self):
        if not self.command_queue:
            self.set_running(False)
            self._set_pipeline_status("Run completed", "success")
            self.statusBar().showMessage("Finished")
            self.refresh_views()
            return

        description, args = self.command_queue.pop(0)
        self.current_command = description
        self.set_running(True)
        self._set_pipeline_status(f"Running {description}", "warning")
        self.statusBar().showMessage(f"Running: {description}")
        self.append_log(f"\n>>> {description}\n$ {sys.executable} {' '.join(args)}\n")
        self.process.start(sys.executable, args)

    def set_running(self, running):
        for button in (
            self.run_all_button,
            self.clear_button,
        ):
            button.setDisabled(running)

    def read_stdout(self):
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self.append_log(data)

    def read_stderr(self):
        data = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
        self.append_log(data)

    def on_process_finished(self, exit_code, exit_status):
        if exit_code != 0:
            self.append_log(f"\nCommand failed with exit code {exit_code}\n")
            self.command_queue.clear()
            self.set_running(False)
            self._set_pipeline_status("Run failed", "danger")
            self.statusBar().showMessage("Failed")
            self.refresh_views()
            return

        self.append_log(f"\nCompleted: {self.current_command}\n")
        self.refresh_views()
        self._run_next_command()

    def on_process_error(self, process_error):
        self.append_log(f"\nProcess error: {process_error}\n")
        self.command_queue.clear()
        self.set_running(False)
        self._set_pipeline_status("Process error", "danger")
        self.statusBar().showMessage("Process error")

    def refresh_results(self):
        for widget in self.image_widgets.values():
            widget.refresh()
        self._sync_chart_previews()

    def _sync_chart_previews(self):
        for scroll in self.chart_scroll_areas.values():
            scroll.sync_preview()

    def refresh_views(self):
        self.refresh_results()
        self.refresh_metadata_summary()

    def reset_views(self):
        for widget in self.image_widgets.values():
            widget.clear_preview()
        self.metadata_text.setPlainText("No model summary loaded yet.\nClick Run All to generate results.")
        self.latest_status_text = "No results loaded yet. Click Run All to generate fresh outputs."
        self.latest_probabilities = {}
        self.latest_regime = None
        self.latest_state = None
        self.latest_price = None
        self.latest_window_share = {}
        self.latest_avg_prob = {}
        self._render_summary()
        self.log_highlights.setPlainText("No run highlights yet.")

    def clear_results(self):
        self.log_output.clear()
        self.reset_views()
        self._set_pipeline_status("View cleared", "info")
        self.statusBar().showMessage("Cleared current view")

    def refresh_metadata_summary(self):
        metadata_path = BASE_DIR / "models" / "metadata.json"
        if not metadata_path.exists():
            self.metadata_text.setPlainText("No model metadata yet.")
            return

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            self.metadata_text.setPlainText("Unable to parse models/metadata.json")
            return

        lines = [f"Model: {metadata.get('model_type', 'unknown')}"]

        metrics = metadata.get("metrics", {})
        if metrics:
            lines.append("")
            lines.append("Metrics")
        for key, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"{key}: {value:.4f}")
            else:
                lines.append(f"{key}: {value}")

        transition_matrix = metadata.get("transition_matrix")
        if transition_matrix:
            lines.append("")
            lines.append("Transition Matrix")
            for idx, row in enumerate(transition_matrix):
                row_text = ", ".join(f"{value:.3f}" for value in row)
                lines.append(f"state {idx}: [{row_text}]")

        postprocess = metadata.get("postprocess", {})
        min_regime_run = postprocess.get("kmeans_min_regime_run")
        if min_regime_run and min_regime_run > 1:
            lines.append("")
            lines.append("Postprocess")
            lines.append(f"KMeans smoothing: min regime run = {min_regime_run} candles")

        state_summary = metadata.get("state_summary") or metadata.get("cluster_summary") or []
        if state_summary:
            lines.append("")
            lines.append("State Summary")
            for item in state_summary:
                state_id = item.get("state", item.get("cluster", "?"))
                lines.append(
                    "state {state}: {name}, 90d={ret:.3f}, vol={vol:.3f}, samples={samples}".format(
                        state=state_id,
                        name=item.get("regime_name", "N/A"),
                        ret=item.get("avg_return_90d", 0.0),
                        vol=item.get("avg_volatility_30d", 0.0),
                        samples=item.get("samples", 0),
                    )
                )

        self.metadata_text.setPlainText("\n".join(lines))


def launch():
    app = QApplication(sys.argv)
    app.setApplicationName("BTC Regime Analysis")

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(launch())
