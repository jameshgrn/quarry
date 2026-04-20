"""Main CLI entry point."""

from pathlib import Path

import click

from georuntime.core.clip import clip_file
from georuntime.core.inspect import inspect_file
from georuntime.core.preview import preview_artifact
from georuntime.core.reproject import reproject_file
from georuntime.registry import Registry


@click.group()
def cli():
    """Personal geospatial runtime."""
    pass


@cli.command()
@click.argument("path")
@click.option("--workspace", "-w", default=None, help="Workspace directory for registry storage.")
@click.option("--quiet", "-q", is_flag=True, help="Output machine-readable summary only.")
def inspect(path: str, workspace: str | None, quiet: bool):
    """Inspect a geospatial file and register it."""
    try:
        artifact = inspect_file(path, workspace)
        if quiet:
            # Machine-readable: type|name|crs|driver|id
            click.echo(
                f"{artifact['artifact_type']}|"
                f"{artifact['name']}|"
                f"{artifact['crs']}|"
                f"{artifact['driver']}|"
                f"{artifact['id']}"
            )
        else:
            click.echo(f"Registered {artifact['artifact_type']} artifact: {artifact['name']}")
            click.echo(f"  ID: {artifact['id']}")
            click.echo(f"  CRS: {artifact['crs']}")
            if artifact.get("extent"):
                ext = artifact["extent"]
                click.echo(
                    f"  Extent: {ext['xmin']:.4f}, {ext['ymin']:.4f}, "
                    f"{ext['xmax']:.4f}, {ext['ymax']:.4f}"
                )
            if artifact["feature_count"] is not None:
                click.echo(f"  Features: {artifact['feature_count']}")
            if artifact["band_count"] is not None:
                click.echo(f"  Bands: {artifact['band_count']}")
            click.echo(f"  Driver: {artifact['driver']}")
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()
    except ImportError as e:
        click.echo(f"Error: Missing dependency - {e}", err=True)
        raise click.Abort()


@cli.command(name="list")
@click.option("--workspace", "-w", default=None, help="Workspace directory for registry storage.")
@click.option(
    "--type",
    "-t",
    "artifact_type",
    default=None,
    help="Filter by artifact type (vector, raster, table, preview, summary).",
)
def list_artifacts(workspace: str | None, artifact_type: str | None):
    """List registered artifacts."""
    reg = Registry(workspace)
    artifacts = reg.list(artifact_type)

    if not artifacts:
        click.echo("No artifacts found.")
        return

    for art in artifacts:
        click.echo(f"{art['id']}  {art['artifact_type']:8}  {art['name']}")


@cli.command()
@click.argument("path")
@click.option("--workspace", "-w", default=None, help="Workspace directory for registry storage.")
@click.option("--quiet", "-q", is_flag=True, help="Output machine-readable summary only.")
def register(path: str, workspace: str | None, quiet: bool):
    """Register an existing geospatial file without inspecting it."""
    try:
        artifact = inspect_file(path, workspace)
        if quiet:
            click.echo(
                f"{artifact['artifact_type']}|"
                f"{artifact['name']}|"
                f"{artifact['crs']}|"
                f"{artifact['driver']}|"
                f"{artifact['id']}"
            )
        else:
            click.echo(f"Registered {artifact['artifact_type']} artifact: {artifact['name']}")
            click.echo(f"  ID: {artifact['id']}")
            click.echo(f"  CRS: {artifact['crs']}")
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.argument("artifact_id")
@click.option("--workspace", "-w", default=None, help="Workspace directory for registry storage.")
def show(artifact_id: str, workspace: str | None):
    """Show details of a specific artifact."""
    reg = Registry(workspace)
    artifact = reg.get(artifact_id)

    if artifact is None:
        click.echo(f"Error: Artifact not found: {artifact_id}", err=True)
        raise click.Abort()

    click.echo(f"Name: {artifact['name']}")
    click.echo(f"Type: {artifact['artifact_type']}")
    click.echo(f"ID: {artifact['id']}")
    click.echo(f"Path: {artifact['path']}")
    click.echo(f"CRS: {artifact['crs']}")
    click.echo(f"Created: {artifact['created_at']}")

    if artifact.get("extent"):
        ext = artifact["extent"]
        click.echo(
            f"Extent: {ext['xmin']:.4f}, {ext['ymin']:.4f}, {ext['xmax']:.4f}, {ext['ymax']:.4f}"
        )
    if artifact["feature_count"] is not None:
        click.echo(f"Features: {artifact['feature_count']}")
    if artifact["band_count"] is not None:
        click.echo(f"Bands: {artifact['band_count']}")
    click.echo(f"Driver: {artifact['driver']}")

    if artifact.get("source_operation"):
        click.echo(f"Source: {artifact['source_operation']}")
    if artifact.get("source_inputs"):
        click.echo(f"Inputs: {artifact['source_inputs']}")
    if artifact.get("metadata"):
        click.echo(f"Metadata: {artifact['metadata']}")


@cli.command()
@click.argument("artifact_id")
@click.argument("target_crs")
@click.option(
    "--output", "-o", default=None, help="Output path (default: <input>_reprojected.<ext>)."
)
@click.option("--workspace", "-w", default=None, help="Workspace directory for registry storage.")
@click.option("--quiet", "-q", is_flag=True, help="Output machine-readable summary only.")
def reproject(
    artifact_id: str, target_crs: str, output: str | None, workspace: str | None, quiet: bool
):
    """Reproject an artifact to a new CRS."""
    reg = Registry(workspace)
    artifact = reg.get(artifact_id)

    if artifact is None:
        click.echo(f"Error: Artifact not found: {artifact_id}", err=True)
        raise click.Abort()

    input_path = Path(artifact["path"])

    # Determine output path
    if output is None:
        suffix = input_path.suffix
        stem = input_path.stem
        output_path = input_path.parent / f"{stem}_reprojected{suffix}"
    else:
        output_path = Path(output)

    try:
        result = reproject_file(input_path, output_path, target_crs)

        # Register the output as a new artifact with lineage
        output_artifact = reg.register(
            {
                "path": str(output_path),
                "artifact_type": artifact["artifact_type"],
                "name": result["name"],
                "crs": result["crs"],
                "band_count": result.get("band_count"),
                "feature_count": result.get("feature_count"),
                "driver": result["driver"],
                "source_operation": "reproject",
                "source_inputs": [artifact_id],
            }
        )

        if quiet:
            click.echo(
                f"{artifact['artifact_type']}|"
                f"{result['name']}|"
                f"{result['crs']}|"
                f"{result['driver']}|"
                f"{output_artifact}"
            )
        else:
            click.echo(f"Reprojected {artifact['name']} → {result['name']}")
            click.echo(f"  Input: {artifact_id}")
            click.echo(f"  Output: {output_artifact}")
            click.echo(f"  New CRS: {result['crs']}")
            click.echo(f"  Output: {output_path}")

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"Error: Reprojection failed - {e}", err=True)
        raise click.Abort()


@cli.command()
@click.argument("artifact_id")
@click.option(
    "--output", "-o", default=None, help="Output PNG path (default: <input>_preview.png)."
)
@click.option("--workspace", "-w", default=None, help="Workspace directory for registry storage.")
@click.option("--quiet", "-q", is_flag=True, help="Output machine-readable summary only.")
def preview(artifact_id: str, output: str | None, workspace: str | None, quiet: bool):
    """Generate a PNG preview of an artifact."""
    reg = Registry(workspace)
    artifact = reg.get(artifact_id)

    if artifact is None:
        click.echo(f"Error: Artifact not found: {artifact_id}", err=True)
        raise click.Abort()

    input_path = Path(artifact["path"])

    # Determine output path
    if output is None:
        output_path = input_path.parent / f"{input_path.stem}_preview.png"
    else:
        output_path = Path(output)

    try:
        result = preview_artifact(input_path, output_path)

        # Register the preview as a new artifact
        preview_id = reg.register(
            {
                "path": str(output_path),
                "artifact_type": "preview",
                "name": result["name"],
                "crs": result["crs"],
                "driver": result["driver"],
                "band_count": None,
                "feature_count": None,
                "source_operation": "preview",
                "source_inputs": [artifact_id],
            }
        )

        if quiet:
            click.echo(
                f"preview|"
                f"{result['name']}|"
                f"{result['driver']}|"
                f"{result['width']}x{result['height']}|"
                f"{preview_id}"
            )
        else:
            click.echo(f"Preview generated: {result['name']}")
            click.echo(f"  Source: {artifact_id}")
            click.echo(f"  Preview: {preview_id}")
            click.echo(f"  Dimensions: {result['width']}x{result['height']}")
            click.echo(f"  Output: {output_path}")

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"Error: Preview generation failed - {e}", err=True)
        raise click.Abort()


@cli.command()
@click.argument("artifact_id")
@click.option("--bounds", "-b", default=None, help="Clip bounds as xmin,ymin,xmax,ymax.")
@click.option("--mask", "-m", default=None, help="Artifact ID to use as clip mask.")
@click.option("--output", "-o", default=None, help="Output path (default: <input>_clipped.<ext>).")
@click.option("--workspace", "-w", default=None, help="Workspace directory for registry storage.")
@click.option("--quiet", "-q", is_flag=True, help="Output machine-readable summary only.")
def clip(
    artifact_id: str,
    bounds: str | None,
    mask: str | None,
    output: str | None,
    workspace: str | None,
    quiet: bool,
):
    """Clip an artifact to bounds or a mask."""
    reg = Registry(workspace)
    artifact = reg.get(artifact_id)

    if artifact is None:
        click.echo(f"Error: Artifact not found: {artifact_id}", err=True)
        raise click.Abort()

    input_path = Path(artifact["path"])

    # Parse bounds if provided
    clip_bounds = None
    if bounds:
        try:
            parts = [float(x.strip()) for x in bounds.split(",")]
            if len(parts) != 4:
                raise ValueError
            clip_bounds = tuple(parts)
        except ValueError:
            click.echo("Error: bounds must be xmin,ymin,xmax,ymax", err=True)
            raise click.Abort()

    # Resolve mask path if mask artifact provided
    mask_path = None
    if mask:
        mask_artifact = reg.get(mask)
        if mask_artifact is None:
            click.echo(f"Error: Mask artifact not found: {mask}", err=True)
            raise click.Abort()
        mask_path = mask_artifact["path"]

    # Determine output path
    if output is None:
        suffix = input_path.suffix
        stem = input_path.stem
        output_path = input_path.parent / f"{stem}_clipped{suffix}"
    else:
        output_path = Path(output)

    try:
        result = clip_file(input_path, output_path, clip_bounds, mask_path)

        # Register the output
        output_id = reg.register(
            {
                "path": str(output_path),
                "artifact_type": artifact["artifact_type"],
                "name": result["name"],
                "crs": result["crs"],
                "band_count": result.get("band_count"),
                "feature_count": result.get("feature_count"),
                "driver": result["driver"],
                "source_operation": "clip",
                "source_inputs": [artifact_id] + ([mask] if mask else []),
            }
        )

        if quiet:
            click.echo(
                f"{artifact['artifact_type']}|"
                f"{result['name']}|"
                f"{result['crs']}|"
                f"{result['driver']}|"
                f"{output_id}"
            )
        else:
            click.echo(f"Clipped {artifact['name']} → {result['name']}")
            click.echo(f"  Input: {artifact_id}")
            click.echo(f"  Output: {output_id}")
            click.echo(f"  Features: {result.get('feature_count', 'N/A')}")
            click.echo(f"  Output: {output_path}")

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"Error: Clip failed - {e}", err=True)
        raise click.Abort()
