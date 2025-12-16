#!/usr/bin/env python3
"""
Export ALL MKs and ALL HSPCs across all slides as SEPARATE page sets.
- mk_page1.html, mk_page2.html, ...
- hspc_page1.html, hspc_page2.html, ...
"""

import json
import argparse
from pathlib import Path
import sys

# Import from the paginated export
sys.path.insert(0, str(Path(__file__).parent))
from export_unified_html_paginated import load_samples

def main():
    parser = argparse.ArgumentParser(description='Export separate MK and HSPC pages for all slides')
    parser.add_argument('--base-dir', type=str, required=True,
                       help='Base directory containing all unified_2pct/SLIDE folders')
    parser.add_argument('--czi-base', type=str, required=True,
                       help='Base directory containing CZI files')
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--samples-per-page', type=int, default=300)
    parser.add_argument('--mk-min-area-um', type=float, default=100,
                       help='Minimum MK area in µm² (must match segmentation filter)')
    parser.add_argument('--mk-max-area-um', type=float, default=2100,
                       help='Maximum MK area in µm² (must match segmentation filter)')

    args = parser.parse_args()

    from pylibCZIrw import czi as pyczi

    base_dir = Path(args.base_dir)
    czi_base = Path(args.czi_base)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all slide directories
    slide_dirs = sorted([d for d in base_dir.glob("2025_11_18_*") if d.is_dir()])

    print(f"\n{'='*60}")
    print(f"Exporting SEPARATE MK and HSPC Pages for ALL Slides")
    print(f"{'='*60}")
    print(f"Found {len(slide_dirs)} slides")
    print(f"Samples per page: {args.samples_per_page}")
    print(f"{'='*60}\n")

    all_mk_samples = []
    all_hspc_samples = []

    # Load samples from all slides
    for slide_dir in slide_dirs:
        slide_name = slide_dir.name
        czi_path = czi_base / f"{slide_name}.czi"

        if not czi_path.exists():
            print(f"⚠ Skipping {slide_name}: CZI not found")
            continue

        print(f"Loading {slide_name}...")

        # Load pixel size
        summary_file = slide_dir / "summary.json"
        pixel_size_um = 0.1725
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)
                if 'pixel_size_um' in summary:
                    pixel_size_um = summary['pixel_size_um'][0]

        # Open CZI
        reader = pyczi.CziReader(str(czi_path))
        scenes = reader.scenes_bounding_rectangle
        if scenes:
            rect = scenes[0]
            x_start, y_start = rect.x, rect.y
        else:
            bbox = reader.total_bounding_box
            x_start, y_start = bbox['X'][0], bbox['Y'][0]

        # Load MK samples
        mk_samples = load_samples(
            slide_dir / "mk" / "tiles",
            reader, x_start, y_start,
            "mk", pixel_size_um, max_samples=None
        )

        # Load HSPC samples
        hspc_samples = load_samples(
            slide_dir / "hspc" / "tiles",
            reader, x_start, y_start,
            "hspc", pixel_size_um, max_samples=None
        )

        reader.close()

        # Add slide name to each sample for tracking
        for s in mk_samples:
            s['slide'] = slide_name
        for s in hspc_samples:
            s['slide'] = slide_name

        all_mk_samples.extend(mk_samples)
        all_hspc_samples.extend(hspc_samples)

        print(f"  ✓ {len(mk_samples)} MKs, {len(hspc_samples)} HSPCs")

    print(f"\n{'='*60}")
    print(f"Total: {len(all_mk_samples)} MKs, {len(all_hspc_samples)} HSPCs")
    print(f"{'='*60}\n")

    # Convert µm² to px² for filtering
    PIXEL_SIZE_UM = 0.1725
    um_to_px_factor = PIXEL_SIZE_UM ** 2  # 0.02975625
    mk_min_px = int(args.mk_min_area_um / um_to_px_factor)
    mk_max_px = int(args.mk_max_area_um / um_to_px_factor)

    # Filter MK cells by size
    print(f"Filtering MK cells by size ({args.mk_min_area_um}-{args.mk_max_area_um} µm² = {mk_min_px}-{mk_max_px} px²)...")
    mk_before = len(all_mk_samples)
    all_mk_samples = [s for s in all_mk_samples if mk_min_px <= s.get('area_px', 0) <= mk_max_px]
    mk_after = len(all_mk_samples)
    print(f"  MK cells: {mk_before} → {mk_after} (removed {mk_before - mk_after})")

    # Sort by area (largest to smallest)
    print("Sorting samples by area (largest to smallest)...")
    all_mk_samples.sort(key=lambda x: x.get('area_um2', 0), reverse=True)
    all_hspc_samples.sort(key=lambda x: x.get('area_um2', 0), reverse=True)

    # Generate separate page sets
    generate_cell_type_pages(all_mk_samples, "mk", output_dir, args.samples_per_page)
    generate_cell_type_pages(all_hspc_samples, "hspc", output_dir, args.samples_per_page)

    # Create unified index
    from create_index_with_export import create_index
    mk_pages = (len(all_mk_samples) + args.samples_per_page - 1) // args.samples_per_page if all_mk_samples else 0
    hspc_pages = (len(all_hspc_samples) + args.samples_per_page - 1) // args.samples_per_page if all_hspc_samples else 0
    create_index(output_dir, len(all_mk_samples), len(all_hspc_samples), mk_pages, hspc_pages)

    print(f"\n{'='*60}")
    print(f"✓ Separate MK and HSPC exports complete!")
    print(f"{'='*60}")
    print(f"Output: {output_dir}")
    print(f"Open: {output_dir / 'index.html'}")
    print(f"{'='*60}\n")


def generate_cell_type_pages(samples, cell_type, output_dir, samples_per_page):
    """Generate separate pages for a single cell type."""

    if not samples:
        print(f"\n⚠ No {cell_type.upper()} samples to export")
        return

    # Split into pages
    pages = [samples[i:i+samples_per_page]
             for i in range(0, len(samples), samples_per_page)]

    total_pages = len(pages)

    print(f"\nGenerating {total_pages} {cell_type.upper()} pages...")

    for page_num in range(1, total_pages + 1):
        page_samples = pages[page_num - 1]

        html = generate_single_type_page_html(
            page_samples, cell_type, page_num, total_pages, output_dir
        )

        html_path = output_dir / f"{cell_type}_page{page_num}.html"
        with open(html_path, 'w') as f:
            f.write(html)

        file_size = html_path.stat().st_size / (1024*1024)
        print(f"  Page {page_num}: {len(page_samples)} samples ({file_size:.1f} MB)")


def generate_single_type_page_html(samples, cell_type, page_num, total_pages, output_dir):
    """Generate HTML for a single cell type page."""

    cell_type_display = "Megakaryocytes (MKs)" if cell_type == "mk" else "HSPCs"

    # Navigation (at top and bottom)
    nav_html = '<div class="page-nav">'
    nav_html += f'<a href="index.html" class="nav-btn">🏠 Home</a>'
    if page_num > 1:
        nav_html += f'<a href="{cell_type}_page{page_num-1}.html" class="nav-btn">← Previous</a>'
    nav_html += f'<span class="page-info">Page {page_num} of {total_pages}</span>'
    if page_num < total_pages:
        nav_html += f'<a href="{cell_type}_page{page_num+1}.html" class="nav-btn">Next →</a>'
    nav_html += '</div>'

    # Generate cards with unique global IDs
    cards_html = ""
    for sample in samples:
        # Construct robust Unique ID (UID)
        # Clean slide name and tile id to ensure valid HTML ID
        slide = sample.get('slide', 'unknown').replace('.', '-')
        tile_id = str(sample.get('tile_id', '0'))
        det_id = sample.get('det_id', 'unknown')
        
        # specific UID format: Slide_Tile_DetID
        uid = f"{slide}_{tile_id}_{det_id}"
        
        area_um2 = sample.get('area_um2', 0)
        area_px = sample.get('area_px', 0)
        img_b64 = sample['image']

        cards_html += f'''
        <div class="card" id="{uid}" data-label="-1">
            <div class="card-img-container">
                <img src="data:image/png;base64,{img_b64}" alt="{det_id}">
            </div>
            <div class="card-info">
                <div>
                    <div class="card-id">{slide} | {tile_id} | {det_id}</div>
                    <div class="card-area">{area_um2:.1f} µm² | {area_px:.0f} px²</div>
                </div>
                <div class="buttons">
                    <button class="btn btn-yes" onclick="setLabel('{cell_type}', '{uid}', 1)">✓</button>
                    <button class="btn btn-unsure" onclick="setLabel('{cell_type}', '{uid}', 2)">?</button>
                    <button class="btn btn-no" onclick="setLabel('{cell_type}', '{uid}', 0)">✗</button>
                </div>
            </div>
        </div>
'''

    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>{cell_type_display} - Page {page_num}/{total_pages}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: monospace; background: #0a0a0a; color: #ddd; }}

        .header {{ background: #111; padding: 12px 20px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #333; position: sticky; top: 0; z-index: 100; }}
        .header h1 {{ font-size: 1.2em; font-weight: normal; }}
        .header-left {{ display: flex; align-items: center; gap: 15px; }}
        .home-btn {{ padding: 8px 16px; background: #1a1a1a; border: 1px solid #333; color: #ddd; text-decoration: none; }}
        .home-btn:hover {{ background: #222; }}
        .stats {{ display: flex; gap: 15px; font-size: 0.85em; }}
        .stat {{ padding: 4px 10px; background: #1a1a1a; border: 1px solid #333; }}
        .stat.positive {{ border-left: 3px solid #4a4; }}
        .stat.negative {{ border-left: 3px solid #a44; }}
        .stat.unsure {{ border-left: 3px solid #da4; }}

        .page-nav {{ text-align: center; padding: 15px; background: #111; border-bottom: 1px solid #333; }}
        .nav-btn {{ display: inline-block; padding: 8px 16px; margin: 0 10px; background: #1a1a1a; color: #ddd; text-decoration: none; border: 1px solid #333; }}
        .nav-btn:hover {{ background: #222; }}
        .page-info {{ margin: 0 20px; }}

        .content {{ padding: 20px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 10px; }}

        .card {{ background: #111; border: 1px solid #333; display: flex; flex-direction: column; }}
        .card-img-container {{ width: 100%; height: 200px; display: flex; align-items: center; justify-content: center; background: #0a0a0a; overflow: hidden; }}
        .card img {{ max-width: 100%; max-height: 100%; object-fit: contain; display: block; margin: auto; }}
        .card-info {{ padding: 8px; display: flex; justify-content: space-between; align-items: center; border-top: 1px solid #333; }}
        .card-id {{ font-size: 0.75em; color: #888; }}
        .card-area {{ font-size: 0.8em; }}

        .buttons {{ display: flex; gap: 4px; }}
        .btn {{ padding: 6px 12px; border: 1px solid #333; background: #1a1a1a; color: #ddd; cursor: pointer; font-family: monospace; font-size: 0.85em; }}
        .btn:hover {{ background: #222; }}

        .card.labeled-yes {{ position: relative; z-index: 10; border: 5px solid #0f0 !important; box-shadow: 0 0 10px #0f0, inset 0 0 10px rgba(0,255,0,0.2); background: #131813 !important; }}
        .card.labeled-no {{ position: relative; z-index: 10; border: 5px solid #f00 !important; box-shadow: 0 0 10px #f00, inset 0 0 10px rgba(255,0,0,0.2); background: #181111 !important; }}
        .card.labeled-unsure {{ position: relative; z-index: 10; border: 5px solid #fa0 !important; box-shadow: 0 0 10px #fa0, inset 0 0 10px rgba(255,170,0,0.2); background: #181611 !important; }}

        .controls {{ position: fixed; bottom: 20px; right: 20px; display: flex; gap: 8px; }}
        .control-btn {{ padding: 10px 20px; border: 1px solid #333; background: #1a1a1a; color: #ddd; cursor: pointer; font-family: monospace; }}
        .control-btn:hover {{ background: #222; }}

        .toast {{ position: fixed; bottom: 70px; right: 20px; background: #222; color: #ddd; padding: 12px 20px; border: 1px solid #333; display: none; }}
    </style>
</head>
<body>
    <div class="header">
        <div class="header-left">
            <a href="index.html" class="home-btn">🏠 Home</a>
            <h1>{cell_type_display} - Page {page_num}/{total_pages}</h1>
        </div>
        <div class="stats">
            <div class="stat">Page: <span id="sample-count">{len(samples)}</span></div>
            <div class="stat positive">✓ <span id="positive-count">0</span></div>
            <div class="stat negative">✗ <span id="negative-count">0</span></div>
            <div class="stat unsure">? <span id="unsure-count">0</span></div>
            <div style="margin: 0 10px; color: #555;">|</div>
            <div class="stat">Global:</div>
            <div class="stat positive">✓ <span id="global-positive-count">0</span></div>
            <div class="stat negative">✗ <span id="global-negative-count">0</span></div>
            <div class="stat unsure">? <span id="global-unsure-count">0</span></div>
        </div>
    </div>

    {nav_html}

    <div class="content">
        <div class="grid">
            {cards_html}
        </div>
    </div>

    {nav_html}

    <div class="controls">
        <button class="control-btn" onclick="clearPageAnnotations()">Clear Page Annotations</button>
    </div>

    <div class="toast" id="toast"></div>

    <script>
        const CELL_TYPE = '{cell_type}';
        const PAGE_NUM = {page_num};
        const STORAGE_KEY = `{cell_type}_labels_page{page_num}`;

        function loadAnnotations() {{
            const stored = localStorage.getItem(STORAGE_KEY);
            if (!stored) return;

            try {{
                const labels = JSON.parse(stored);
                for (const [uid, label] of Object.entries(labels)) {{
                    const card = document.getElementById(uid);
                    if (card && label !== -1) {{
                        card.dataset.label = label;
                        card.classList.remove('labeled-yes', 'labeled-no', 'labeled-unsure');
                        if (label == 1) {{
                            card.classList.add('labeled-yes');
                        }} else if (label == 2) {{
                            card.classList.add('labeled-unsure');
                        }} else {{
                            card.classList.add('labeled-no');
                        }}
                    }}
                }}
                updateStats();
            }} catch(e) {{
                console.error('Failed to load annotations:', e);
            }}
        }}

        function setLabel(cellType, uid, label) {{
            const card = document.getElementById(uid);
            if (!card) {{
                console.error('Card not found:', uid);
                return;
            }}

            card.dataset.label = label;

            // Remove all label classes first
            card.classList.remove('labeled-yes', 'labeled-no', 'labeled-unsure');

            // Add new label class
            if (label == 1) {{
                card.classList.add('labeled-yes');
            }} else if (label == 2) {{
                card.classList.add('labeled-unsure');
            }} else {{
                card.classList.add('labeled-no');
            }}

            saveAnnotations();
            updateStats();
        }}

        function saveAnnotations() {{
            const labels = {{}};
            document.querySelectorAll('.card').forEach(card => {{
                const uid = card.id;
                const label = parseInt(card.dataset.label);
                if (label !== -1) {{
                    labels[uid] = label;
                }}
            }});

            localStorage.setItem(STORAGE_KEY, JSON.stringify(labels));
        }}

        function updateStats() {{
            // Page-level counts
            let positiveCount = 0;
            let negativeCount = 0;
            let unsureCount = 0;

            document.querySelectorAll('.card').forEach(card => {{
                const label = parseInt(card.dataset.label);
                if (label === 1) positiveCount++;
                else if (label === 0) negativeCount++;
                else if (label === 2) unsureCount++;
            }});

            document.getElementById('positive-count').textContent = positiveCount;
            document.getElementById('negative-count').textContent = negativeCount;
            document.getElementById('unsure-count').textContent = unsureCount;

            // Global counts across all pages
            let globalPositive = 0;
            let globalNegative = 0;
            let globalUnsure = 0;

            for (let page = 1; page <= {total_pages}; page++) {{
                const key = `{cell_type}_labels_page${{page}}`;
                const pageLabels = localStorage.getItem(key);
                if (pageLabels) {{
                    try {{
                        const labels = JSON.parse(pageLabels);
                        for (const [uid, label] of Object.entries(labels)) {{
                            if (label === 1) globalPositive++;
                            else if (label === 0) globalNegative++;
                            else if (label === 2) globalUnsure++;
                        }}
                    }} catch(e) {{}}
                }}
            }}

            document.getElementById('global-positive-count').textContent = globalPositive;
            document.getElementById('global-negative-count').textContent = globalNegative;
            document.getElementById('global-unsure-count').textContent = globalUnsure;
        }}

        function clearPageAnnotations() {{
            if (!confirm('Clear all annotations on this page?')) return;

            localStorage.removeItem(STORAGE_KEY);
            document.querySelectorAll('.card').forEach(card => {{
                card.dataset.label = -1;
                card.className = 'card';
            }});
            updateStats();
            showToast('Page annotations cleared');
        }}

        function showToast(message) {{
            const toast = document.getElementById('toast');
            toast.textContent = message;
            toast.style.display = 'block';
            setTimeout(() => {{ toast.style.display = 'none'; }}, 3000);
        }}

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowLeft' && {page_num} > 1) {{
                window.location.href = '{cell_type}_page{page_num-1}.html';
            }} else if (e.key === 'ArrowRight' && {page_num} < {total_pages}) {{
                window.location.href = '{cell_type}_page{page_num+1}.html';
            }}
        }});

        loadAnnotations();
    </script>
</body>
</html>'''

    return html


def create_index_OLD(output_dir, total_mks, total_hspcs, samples_per_page):
    """OLD - Create unified index page."""

    mk_pages = (total_mks + samples_per_page - 1) // samples_per_page if total_mks > 0 else 0
    hspc_pages = (total_hspcs + samples_per_page - 1) // samples_per_page if total_hspcs > 0 else 0

    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Combined MK+HSPC Review (Separate Sets)</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: monospace; background: #0a0a0a; color: #ddd; padding: 20px; }}

        .header {{ background: #111; padding: 20px; border: 1px solid #333; margin-bottom: 20px; text-align: center; }}
        h1 {{ font-size: 1.5em; font-weight: normal; margin-bottom: 15px; }}

        .stats {{ display: flex; justify-content: center; gap: 30px; margin: 20px 0; }}
        .stat {{ padding: 15px 30px; background: #1a1a1a; border: 1px solid #333; }}
        .stat .number {{ display: block; font-size: 2em; margin-top: 10px; }}

        .note {{ text-align: center; margin: 20px 0; padding: 15px; background: #111; border: 1px solid #333; border-left: 3px solid #555; }}

        .section {{ margin: 40px 0; }}
        .section h2 {{ font-size: 1.3em; margin-bottom: 15px; padding: 10px; background: #111; border: 1px solid #333; border-left: 3px solid #555; }}

        .controls {{ text-align: center; margin: 30px 0; }}
        .btn {{ padding: 15px 30px; background: #1a1a1a; border: 1px solid #333; color: #ddd; cursor: pointer; font-family: monospace; font-size: 1.1em; margin: 10px; text-decoration: none; display: inline-block; }}
        .btn:hover {{ background: #222; }}
        .btn-primary {{ border-color: #4a4; color: #4a4; }}
        .btn-clear {{ border-color: #a44; color: #a44; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Combined MK + HSPC Review</h1>
        <p style="color: #888;">All slides combined - Phase 1 (2% sample) - Separate annotation sets</p>

        <div class="stats">
            <div class="stat">
                <span>Total MKs</span>
                <span class="number">{total_mks:,}</span>
            </div>
            <div class="stat">
                <span>Total HSPCs</span>
                <span class="number">{total_hspcs:,}</span>
            </div>
            <div class="stat">
                <span>MK Annotations</span>
                <span class="number" id="mk-annotation-count">0</span>
            </div>
            <div class="stat">
                <span>HSPC Annotations</span>
                <span class="number" id="hspc-annotation-count">0</span>
            </div>
        </div>

        <div class="note">
            ⚠ MKs and HSPCs are now <strong>completely separate</strong> - each with their own pages and annotations
        </div>
    </div>

    <div class="section">
        <h2>🔴 Megakaryocytes (MKs)</h2>
        <div class="controls">
            <a href="mk_page1.html" class="btn btn-primary">Review MKs →</a>
            <span style="margin: 0 20px; color: #888;">{mk_pages} pages • ~{samples_per_page} per page</span>
        </div>
    </div>

    <div class="section">
        <h2>🔵 HSPCs</h2>
        <div class="controls">
            <a href="hspc_page1.html" class="btn btn-primary">Review HSPCs →</a>
            <span style="margin: 0 20px; color: #888;">{hspc_pages} pages • ~{samples_per_page} per page</span>
        </div>
    </div>

    <div class="controls">
        <button class="btn btn-clear" onclick="clearAllAnnotations()">Clear All Annotations</button>
    </div>

    <script>
        function updateAnnotationCounts() {{
            let mkAnnotations = 0;
            let hspcAnnotations = 0;

            const allKeys = Object.keys(localStorage);

            for (const key of allKeys) {{
                if (key.startsWith('mk_labels_page')) {{
                    try {{
                        const labels = JSON.parse(localStorage.getItem(key));
                        mkAnnotations += Object.keys(labels).length;
                    }} catch(e) {{}}
                }} else if (key.startsWith('hspc_labels_page')) {{
                    try {{
                        const labels = JSON.parse(localStorage.getItem(key));
                        hspcAnnotations += Object.keys(labels).length;
                    }} catch(e) {{}}
                }}
            }}

            document.getElementById('mk-annotation-count').textContent = mkAnnotations.toLocaleString();
            document.getElementById('hspc-annotation-count').textContent = hspcAnnotations.toLocaleString();
        }}

        function clearAllAnnotations() {{
            if (!confirm('Clear ALL annotations for both MKs and HSPCs? This cannot be undone.')) return;

            const keysToDelete = Object.keys(localStorage).filter(key =>
                key.startsWith('mk_labels_') || key.startsWith('hspc_labels_')
            );

            keysToDelete.forEach(key => localStorage.removeItem(key));

            alert(`Cleared ${{keysToDelete.length}} annotation pages.`);
            location.reload();
        }}

        updateAnnotationCounts();
    </script>
</body>
</html>'''

    with open(output_dir / 'index.html', 'w') as f:
        f.write(html)


if __name__ == '__main__':
    main()
