from __future__ import annotations

import argparse
from pathlib import Path

from .config import PipelineConfig, WindowMappingConfig
from .data import list_images
from .evaluation import (
    evaluate_leave_one_out,
    format_evaluation_report,
    write_confusion_csv,
    write_rows_csv,
)
from .pipeline import ScaleEstimationPipeline
from .recommendation import (
    format_recommendation_report,
    recommend_windows,
    write_recommendations_csv,
    write_refer_summary_csv,
)
from .visualization import (
    format_visualization_report,
    generate_window_visualizations,
    write_visualization_summary_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rate-identification",
        description="Continuous scale estimation for adaptive sliding windows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit", help="Build an unsupervised scale library from images.")
    fit_parser.add_argument("input", type=Path, help="Input image folder or a single image file.")
    fit_parser.add_argument("--output", type=Path, required=True, help="Output joblib model path.")
    fit_parser.add_argument(
        "--extractor",
        default="robust_handcrafted",
        choices=["robust_handcrafted", "dinov2_timm"],
        help="Feature extractor backend.",
    )
    fit_parser.add_argument("--pca-components", type=int, default=64)
    fit_parser.add_argument("--knn-neighbors", type=int, default=5)
    fit_parser.add_argument("--cluster-count", type=int, default=6)
    fit_parser.add_argument("--resize-long-edge", type=int, default=1024)
    fit_parser.add_argument("--disable-multiview", action="store_true")

    infer_parser = subparsers.add_parser("infer", help="Infer continuous scale and window size.")
    infer_parser.add_argument("input", type=Path, help="Input image path.")
    infer_parser.add_argument("--model", type=Path, required=True, help="Trained joblib model path.")
    infer_parser.add_argument("--w-min", type=int, default=None)
    infer_parser.add_argument("--w-max", type=int, default=None)
    infer_parser.add_argument("--alpha", type=float, default=None)
    infer_parser.add_argument("--beta", type=float, default=None)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate magnification accuracy on labeled folders.")
    eval_parser.add_argument("input", type=Path, help="Root directory containing per-magnification subfolders.")
    eval_parser.add_argument(
        "--extractor",
        default="robust_handcrafted",
        choices=["robust_handcrafted", "dinov2_timm"],
        help="Feature extractor backend.",
    )
    eval_parser.add_argument("--pca-components", type=int, default=64)
    eval_parser.add_argument("--knn-neighbors", type=int, default=5)
    eval_parser.add_argument("--cluster-count", type=int, default=6)
    eval_parser.add_argument("--resize-long-edge", type=int, default=1024)
    eval_parser.add_argument("--disable-multiview", action="store_true")
    eval_parser.add_argument("--csv", type=Path, default=None, help="Optional output CSV for per-image predictions.")
    eval_parser.add_argument("--confusion-csv", type=Path, default=None, help="Optional output CSV for confusion matrix.")

    recommend_parser = subparsers.add_parser("recommend-windows", help="Recommend sliding window sizes to match the refer field of view.")
    recommend_parser.add_argument("--labeled-root", type=Path, default=Path("data/images"), help="Root directory containing labeled magnification folders.")
    recommend_parser.add_argument("--refer-root", type=Path, default=Path("data/images/refer"), help="Reference image directory used by the segmentation training set.")
    recommend_parser.add_argument(
        "--extractor",
        default="dinov2_timm",
        choices=["robust_handcrafted", "dinov2_timm"],
        help="Feature extractor backend.",
    )
    recommend_parser.add_argument("--pca-components", type=int, default=64)
    recommend_parser.add_argument("--knn-neighbors", type=int, default=5)
    recommend_parser.add_argument("--cluster-count", type=int, default=6)
    recommend_parser.add_argument("--resize-long-edge", type=int, default=1024)
    recommend_parser.add_argument("--disable-multiview", action="store_true")
    recommend_parser.add_argument("--align-to", type=int, default=32, help="Round recommended windows to this multiple.")
    recommend_parser.add_argument("--proxy-stat", default="median", choices=["median", "p75", "max"], help="Statistic used to aggregate refer field-of-view proxies.")
    recommend_parser.add_argument("--square-basis", default="short", choices=["short", "long"], help="Use refer short or long side to derive square windows.")
    recommend_parser.add_argument("--csv", type=Path, default=None, help="Optional CSV for per-magnification window recommendations.")
    recommend_parser.add_argument("--refer-csv", type=Path, default=None, help="Optional CSV summarizing refer image predictions.")

    visualize_parser = subparsers.add_parser("visualize-windows", help="Generate one patch visualization per magnification.")
    visualize_parser.add_argument("--labeled-root", type=Path, default=Path("data/images"), help="Root directory containing labeled magnification folders.")
    visualize_parser.add_argument("--recommendations-csv", type=Path, default=Path("artifacts/window_recommendations.csv"), help="CSV containing recommended window sizes.")
    visualize_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/window_visualizations"), help="Directory for generated patch and preview images.")
    visualize_parser.add_argument("--display-size", type=int, default=768, help="Output patch visualization size in pixels.")
    visualize_parser.add_argument("--csv", type=Path, default=None, help="Optional summary CSV for generated visualization files.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "fit":
        config = PipelineConfig(
            resize_long_edge=args.resize_long_edge,
            pca_components=args.pca_components,
            knn_neighbors=args.knn_neighbors,
            cluster_count=args.cluster_count,
            extractor_name=args.extractor,
            use_multiview_inference=not args.disable_multiview,
        )
        image_paths = list_images(args.input)
        if not image_paths:
            raise SystemExit(f"No images found under: {args.input}")
        pipeline = ScaleEstimationPipeline(config=config).fit(image_paths)
        pipeline.save(args.output)
        print(f"saved_model={args.output}")
        print(f"fitted_images={len(image_paths)}")
        return

    if args.command == "infer":
        pipeline = ScaleEstimationPipeline.load(args.model)
        if any(value is not None for value in [args.w_min, args.w_max, args.alpha, args.beta]):
            pipeline.config.window = WindowMappingConfig(
                w_min=args.w_min or pipeline.config.window.w_min,
                w_max=args.w_max or pipeline.config.window.w_max,
                alpha=args.alpha or pipeline.config.window.alpha,
                beta=args.beta or pipeline.config.window.beta,
            )
        result = pipeline.predict(args.input)
        print(f"predicted_scale={result['scale']:.6f}")
        print(f"window_size={result['window_size']}")
        return

    if args.command == "evaluate":
        config = PipelineConfig(
            resize_long_edge=args.resize_long_edge,
            pca_components=args.pca_components,
            knn_neighbors=args.knn_neighbors,
            cluster_count=args.cluster_count,
            extractor_name=args.extractor,
            use_multiview_inference=not args.disable_multiview,
        )
        results = evaluate_leave_one_out(args.input, config=config)
        print(format_evaluation_report(results))
        if args.csv is not None:
            write_rows_csv(args.csv, results["rows"])
            print(f"saved_csv={args.csv}")
        if args.confusion_csv is not None:
            write_confusion_csv(args.confusion_csv, results["labels"], results["confusion_matrix"])
            print(f"saved_confusion_csv={args.confusion_csv}")
        return

    if args.command == "recommend-windows":
        config = PipelineConfig(
            resize_long_edge=args.resize_long_edge,
            pca_components=args.pca_components,
            knn_neighbors=args.knn_neighbors,
            cluster_count=args.cluster_count,
            extractor_name=args.extractor,
            use_multiview_inference=not args.disable_multiview,
        )
        results = recommend_windows(
            labeled_root=args.labeled_root,
            refer_root=args.refer_root,
            config=config,
            align_to=args.align_to,
            proxy_stat=args.proxy_stat,
            square_basis=args.square_basis,
        )
        print(format_recommendation_report(results))
        if args.csv is not None:
            write_recommendations_csv(args.csv, results["recommendations"])
            print(f"saved_csv={args.csv}")
        if args.refer_csv is not None:
            write_refer_summary_csv(args.refer_csv, results["refer_rows"])
            print(f"saved_refer_csv={args.refer_csv}")
        return

    if args.command == "visualize-windows":
        rows = generate_window_visualizations(
            labeled_root=args.labeled_root,
            recommendations_csv=args.recommendations_csv,
            output_dir=args.output_dir,
            display_size=args.display_size,
        )
        print(format_visualization_report(rows))
        if args.csv is not None:
            write_visualization_summary_csv(args.csv, rows)
            print(f"saved_csv={args.csv}")
        return

    raise SystemExit("Unsupported command")


if __name__ == "__main__":
    main()
