import argparse
import logging
from pathlib import Path
from typing import Tuple
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

DEFAULT_MS_PATH = "path/to/your/3m_multispectral.tif"
DEFAULT_PAN_PATH = "path/to/your/1m_panchromatic.tif"
DEFAULT_OUTPUT_PATH = "path/to/your/1m_output_sharpened.tif"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("PansharpeningPipeline")


class SatellitePansharpeningPipeline:
    def __init__(self, ms_path: Path, pan_path: Path, output_path: Path, block_size: int = 2048):
        self.ms_path = Path(ms_path)
        self.pan_path = Path(pan_path)
        self.output_path = Path(output_path)
        self.block_size = block_size

    def _validate_inputs(self) -> None:
        if not self.ms_path.exists():
            raise FileNotFoundError(f"Multispectral raster missing: {self.ms_path}")
        if not self.pan_path.exists():
            raise FileNotFoundError(f"Panchromatic raster missing: {self.pan_path}")

    @staticmethod
    def _compute_weights(ms_block: np.ndarray, pan_block: np.ndarray) -> np.ndarray:
        nbands = ms_block.shape[0]
        flat_ms = ms_block.reshape(nbands, -1)
        flat_pan = pan_block.flatten()

        cov_matrix = np.cov(flat_ms)
        cov_pan = np.array([np.cov(flat_ms[i], flat_pan)[0, 1] for i in range(nbands)])

        try:
            weights = np.linalg.solve(cov_matrix, cov_pan)
        except np.linalg.LinAlgError:
            weights = np.full(nbands, 1.0 / nbands)

        weights = np.maximum(weights, 0)
        total = np.sum(weights)
        return weights / total if total > 0 else np.full(nbands, 1.0 / nbands)

    def process(self) -> None:
        self._validate_inputs()
        logger.info("Opening dataset metadata...")

        with rasterio.open(self.pan_path) as pan_ds, rasterio.open(self.ms_path) as ms_ds:
            pan_meta = pan_ds.meta.copy()
            pan_width = pan_ds.width
            pan_height = pan_ds.height
            num_ms_bands = ms_ds.count

            out_meta = pan_meta.copy()
            out_meta.update({
                "count": num_ms_bands,
                "dtype": "uint16",
                "nodata": ms_ds.nodata if ms_ds.nodata is not None else 0,
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
                "compress": "lzw"
            })

            logger.info(f"Target dimensions: {pan_width}x{pan_height} at 1m resolution.")
            logger.info(f"Processing 4 bands (RGB + NIR) in block size {self.block_size}x{self.block_size}...")

            with rasterio.open(self.output_path, "w", **out_meta) as out_ds:
                for y in range(0, pan_height, self.block_size):
                    for x in range(0, pan_width, self.block_size):
                        w_width = min(self.block_size, pan_width - x)
                        w_height = min(self.block_size, pan_height - y)
                        pan_window = Window(x, y, w_width, w_height)

                        pan_block = pan_ds.read(1, window=pan_window).astype(np.float32)

                        pan_win_transform = rasterio.windows.transform(pan_window, pan_ds.transform)
                        ms_window = rasterio.windows.from_bounds(
                            *rasterio.windows.bounds(pan_window, pan_ds.transform),
                            transform=ms_ds.transform
                        )

                        ms_upsampled = ms_ds.read(
                            out_shape=(num_ms_bands, w_height, w_width),
                            window=ms_window,
                            resampling=Resampling.bilinear
                        ).astype(np.float32)

                        weights = self._compute_weights(ms_upsampled, pan_block)
                        simulated_pan = np.zeros((w_height, w_width), dtype=np.float32)
                        for b in range(num_ms_bands):
                            simulated_pan += weights[b] * ms_upsampled[b]

                        pan_diff = pan_block - simulated_pan

                        sharpened_block = np.zeros_like(ms_upsampled)
                        for b in range(num_ms_bands):
                            sharpened_block[b] = np.clip(
                                ms_upsampled[b] + pan_diff,
                                0,
                                65535
                            )

                        out_ds.write(sharpened_block.astype(np.uint16), window=pan_window)

        logger.info(f"Processing complete. Output written to: {self.output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="High-resolution Bilinear Upsampling and Gram-Schmidt Pansharpening for Multispectral Imagery."
    )
    parser.add_argument("--ms", type=str, default=DEFAULT_MS_PATH, help="Path to 4-band 3m Multispectral Image")
    parser.add_argument("--pan", type=str, default=DEFAULT_PAN_PATH, help="Path to 1-band 1m Panchromatic Image")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH, help="Path for output 1m 4-band Image")
    parser.add_argument("--block-size", type=int, default=2048, help="Window tile size for memory control")

    args = parser.parse_args()

    pipeline = SatellitePansharpeningPipeline(
        ms_path=args.ms,
        pan_path=args.pan,
        output_path=args.output,
        block_size=args.block_size
    )
    pipeline.process()


if __name__ == "__main__":
    main()

