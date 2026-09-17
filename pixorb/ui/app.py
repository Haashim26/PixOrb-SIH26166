"""PixOrb desktop GUI (PySide6)."""
import json, os, sys, threading
from pathlib import Path
from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtGui import QPixmap, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox, QLabel,
    QPushButton, QVBoxLayout, QHBoxLayout, QTabWidget, QTextEdit, QProgressBar,
    QGroupBox, QGridLayout, QScrollArea, QFrame
)
from ..pipeline import run


class Worker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, ref, src, out):
        super().__init__()
        self.ref, self.src, self.out = ref, src, out

    def execute(self):
        try:
            self.finished.emit(run(self.ref, self.src, self.out))
        except Exception as e:
            self.failed.emit(str(e))


class ImageView(QLabel):
    def __init__(self, placeholder):
        super().__init__(placeholder)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(420, 300)
        self.setStyleSheet(
            "background:#111820;border:1px solid #31404d;border-radius:6px;"
            "color:#9aa8b5;"
        )
        self._pix = None

    def set_image(self, path):
        pix = QPixmap(str(path))
        if pix.isNull():
            self.setText(f"Unable to display:\n{path}")
            return
        self._pix = pix
        self._refresh()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self):
        if self._pix:
            self.setPixmap(self._pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


class MetricCard(QFrame):
    def __init__(self, title, value, detail=""):
        super().__init__()
        self.setStyleSheet(
            "QFrame{background:#111a22;border:1px solid #2b3a47;border-radius:8px;}"
            "QLabel.title{color:#8fa1b2;font-size:11px;}"
            "QLabel.value{color:#eef4f8;font-size:20px;font-weight:600;}"
            "QLabel.detail{color:#748797;font-size:10px;}"
        )
        l = QVBoxLayout(self)
        l.setContentsMargins(12, 9, 12, 9)
        t = QLabel(title); t.setProperty("class", "title")
        v = QLabel(value); v.setProperty("class", "value")
        d = QLabel(detail); d.setProperty("class", "detail")
        l.addWidget(t); l.addWidget(v); l.addWidget(d)


class PixOrbApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ref = None
        self.src = None
        self.output = None
        self.worker = None
        self.setWindowTitle("PixOrb — SIH26166 Lunar Image Correspondence")
        self.resize(1500, 920)
        self.setStyleSheet("""
            QMainWindow { background:#0b1117; color:#e8edf2; }
            QWidget { color:#e8edf2; font-family:'Segoe UI'; font-size:13px; }
            QGroupBox { border:1px solid #263440; border-radius:8px; margin-top:12px; padding:10px; }
            QGroupBox::title { subcontrol-origin:margin; left:12px; padding:0 5px; color:#8fc7ff; }
            QPushButton { background:#1d2a36; border:1px solid #3a4d5e; border-radius:6px; padding:9px 14px; }
            QPushButton:hover { background:#263746; }
            QPushButton:disabled { color:#65727e; }
            QTabWidget::pane { border:1px solid #263440; }
            QTabBar::tab { padding:9px 18px; }
            QTabBar::tab:selected { background:#1d2a36; }
            QTextEdit { background:#101820; border:1px solid #263440; }
            QScrollArea { border:none; background:#0b1117; }
        """)

        root = QWidget(); self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(10, 8, 10, 10)

        title = QLabel("PIXORB  |  SIH26166")
        title.setFont(QFont("Segoe UI", 24, QFont.Bold))
        layout.addWidget(title)
        sub = QLabel("Multi-modal, sun-angle and scale-invariant lunar image correspondence")
        sub.setStyleSheet("color:#9aa8b5;font-size:14px;")
        layout.addWidget(sub)

        controls = QHBoxLayout()
        self.ref_btn = QPushButton("Load Reference Image"); self.ref_btn.clicked.connect(self.load_ref); controls.addWidget(self.ref_btn)
        self.src_btn = QPushButton("Load Source Image"); self.src_btn.clicked.connect(self.load_src); controls.addWidget(self.src_btn)
        self.run_btn = QPushButton("REGISTER IMAGES"); self.run_btn.clicked.connect(self.register); controls.addWidget(self.run_btn)
        self.open_btn = QPushButton("Open Results"); self.open_btn.clicked.connect(self.open_results); controls.addWidget(self.open_btn)
        layout.addLayout(controls)

        self.status = QLabel("Load a reference image and a source image to begin.")
        self.status.setStyleSheet("color:#9aa8b5;")
        layout.addWidget(self.status)
        self.progress = QProgressBar(); self.progress.setRange(0, 0); self.progress.hide(); layout.addWidget(self.progress)

        tabs = QTabWidget(); layout.addWidget(tabs, 1)

        # Input
        input_tab = QWidget(); il = QGridLayout(input_tab)
        self.ref_view = ImageView("Reference image"); self.src_view = ImageView("Source image")
        il.addWidget(QLabel("REFERENCE"), 0, 0); il.addWidget(QLabel("SOURCE"), 0, 1)
        il.addWidget(self.ref_view, 1, 0); il.addWidget(self.src_view, 1, 1)
        tabs.addTab(input_tab, "Input")

        # Correspondences
        match_tab = QWidget(); ml = QVBoxLayout(match_tab)
        self.match_info = QLabel("Final verified correspondences")
        self.match_info.setStyleSheet("color:#9aa8b5;padding:4px;")
        ml.addWidget(self.match_info)
        self.match_view = ImageView("Correspondence visualization")
        ml.addWidget(self.match_view, 1)
        tabs.addTab(match_tab, "Correspondences")

        # Registered output: separate scientific views, avoiding the old hard-edged
        # square-only presentation.
        out_tab = QWidget(); ol = QVBoxLayout(out_tab)
        self.output_tabs = QTabWidget()
        self.registered_view = ImageView("Registered source")
        self.overlay_view = ImageView("Reference + registered source overlay")
        self.diff_view = ImageView("Valid-overlap difference map")
        self.output_tabs.addTab(self.overlay_view, "Overlay")
        self.output_tabs.addTab(self.registered_view, "Registered Source")
        self.output_tabs.addTab(self.diff_view, "Difference Map")
        ol.addWidget(self.output_tabs, 1)
        self.output_note = QLabel("Overlay: reference + registered source inside valid overlap. Difference map is an intensity diagnostic; geometric RMSE is reported separately in Metrics.")
        self.output_note.setStyleSheet("color:#7f91a0;padding:4px;")
        ol.addWidget(self.output_note)
        tabs.addTab(out_tab, "Registered Output")

        # Metrics
        met_tab = QWidget(); tl = QVBoxLayout(met_tab)
        self.cards = QGridLayout(); tl.addLayout(self.cards)
        self.metrics = QTextEdit(); self.metrics.setReadOnly(True); tl.addWidget(self.metrics, 1)
        tabs.addTab(met_tab, "Metrics")
        self.tabs = tabs

    def load_ref(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select reference image", "", "Images (*.jpg *.jpeg *.png *.tif *.tiff *.tif);;All files (*)")
        if p:
            self.ref = p; self.ref_view.set_image(p); self.status.setText(f"Reference: {p}")

    def load_src(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select source image", "", "Images (*.jpg *.jpeg *.png *.tif *.tiff);;All files (*)")
        if p:
            self.src = p; self.src_view.set_image(p); self.status.setText(f"Source: {p}")

    def register(self):
        if not (self.ref and self.src):
            QMessageBox.warning(self, "PixOrb", "Load both reference and source images first.")
            return
        out = QFileDialog.getExistingDirectory(self, "Choose output directory")
        if not out:
            return
        self.run_btn.setEnabled(False); self.ref_btn.setEnabled(False); self.src_btn.setEnabled(False)
        self.progress.show(); self.status.setText("Running coarse-to-fine PixOrb registration…")
        self.worker = Worker(self.ref, self.src, out)
        self.worker.finished.connect(self.finish); self.worker.failed.connect(self.failed)
        threading.Thread(target=self.worker.execute, daemon=True).start()

    def _clear_cards(self):
        while self.cards.count():
            item = self.cards.takeAt(0)
            w = item.widget()
            if w: w.deleteLater()

    def _set_metrics(self, m):
        self._clear_cards()
        cards = [
            ("RMSE", f"{m.get('rmse_px', float('nan')):.3f} px", "geometric reprojection error"),
            ("Mean error", f"{m.get('mean_reprojection_error_px', float('nan')):.3f} px", "mean residual"),
            ("Final matches", str(m.get('final_match_count', 0)), "final verified correspondences"),
            ("Initial inliers", f"{100*m.get('initial_ransac_inlier_ratio', 0):.2f}%", f"{m.get('initial_ransac_inlier_count', 0)} / {m.get('deduplicated_candidates', 0)}"),
            ("Final verification", f"{100*m.get('final_ransac_inlier_ratio', 0):.2f}%", "after ANMS + final RANSAC"),
            ("Spatial entropy", f"{m.get('spatial_entropy_normalized', 0):.4f}", "normalized distribution score"),
            ("Overlap", f"{m.get('overlap_percent', 0):.2f}%", "valid geometric overlap"),
            ("Grid CV", f"{m.get('grid_count_cv', 0):.3f}", "lower = more even counts"),
        ]
        for i, (a,b,c) in enumerate(cards):
            self.cards.addWidget(MetricCard(a,b,c), i//4, i%4)
        self.metrics.setPlainText(json.dumps(m, indent=2, default=str))

    def finish(self, r):
        self.output = r
        self.progress.hide(); self.run_btn.setEnabled(True); self.ref_btn.setEnabled(True); self.src_btn.setEnabled(True)
        m = r['metrics']; self.status.setText(f"Registration complete — {m.get('backend', 'unknown backend')}")
        self._set_metrics(m)
        od = Path(r['output_dir'])
        self.match_view.set_image(od / 'matches.png')
        self.registered_view.set_image(od / 'registered.png')
        self.overlay_view.set_image(od / 'overlay.png')
        self.diff_view.set_image(od / 'difference_map.png')
        self.match_info.setText(
            f"Final verified correspondences: {m.get('final_match_count', 0)}  |  "
            f"Initial RANSAC inliers: {m.get('initial_ransac_inlier_count', 0)}  |  "
            f"Final verification: {100*m.get('final_ransac_inlier_ratio', 0):.2f}%"
        )
        self.tabs.setCurrentIndex(2)

    def failed(self, msg):
        self.progress.hide(); self.run_btn.setEnabled(True); self.ref_btn.setEnabled(True); self.src_btn.setEnabled(True)
        self.status.setText("Registration failed.")
        QMessageBox.critical(self, "PixOrb error", msg)

    def open_results(self):
        if self.output:
            os.startfile(self.output['output_dir']) if hasattr(os, 'startfile') else None
        else:
            QMessageBox.information(self, "PixOrb", "Run a registration first.")


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    win = PixOrbApp(); win.show()
    return app.exec()
