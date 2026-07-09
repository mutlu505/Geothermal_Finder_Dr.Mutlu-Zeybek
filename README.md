# Geothermal_Finder_Dr.Mutlu-Zeybek
Geothermal_Finder_Dr.Mutlu Zeybek
"""
ZEYBEK-2 Model: A Rule-Based Expert System for Systematic, Geometry-Driven Targeting 
of Fault-Controlled Geothermal Reservoirs

Author: Mutlu ZEYBEK
Affiliation: Muğla Metropolitan Municipality, Muğla, Turkey


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Polygon
import numpy as np


def draw_zeybek2_with_coordinates():
    """
    Draws Figure 1 from the ZEYBEK-2 Model paper with full coordinate system
    showing the geometric configuration and explicit coordinate labels.
    Includes North Arrow and Scale Bar.
    """

    # Create figure with specific size
    fig, ax = plt.subplots(figsize=(16, 10))

    # Define coordinates for the geological blocks
    # Following the model: F4 | L1 | F1 | L2 | F2 | L3 | F3 | L2 | F4
    x0, x1, x2, x3, x4, x5, x6 = 0, 2, 4, 5.5, 7, 9, 11
    y0, y1, y2 = 0, 3.5, 7

    # Define blocks with colors and labels
    blocks = [
        # Left Heat Source Rock (L1) - between F4 and F1
        {
            "x": x0,
            "y": y0,
            "width": x1 - x0,
            "height": y2 - y0,
            "label": "L1",
            "sub": "Heat Source Rock",
            "color": "#FF6B6B",
            "hatch": "///",
            "alpha": 0.7,
        },
        # Left Reservoir Rock (L2) - between F1 and F2
        {
            "x": x1,
            "y": y0,
            "width": x2 - x1,
            "height": y2 - y0,
            "label": "L2",
            "sub": "Reservoir Rock",
            "color": "#4ECDC4",
            "hatch": "...",
            "alpha": 0.7,
        },
        # Cap Rock (L3) - TARGET ZONE between F2 and F3
        {
            "x": x2,
            "y": y0,
            "width": x4 - x2,
            "height": y2 - y0,
            "label": "L3",
            "sub": "Cap Rock (Seal)",
            "color": "#FFE66D",
            "hatch": "xxx",
            "alpha": 0.7,
        },
        # Right Reservoir Rock (L2) - between F3 and F4
        {
            "x": x4,
            "y": y0,
            "width": x5 - x4,
            "height": y2 - y0,
            "label": "L2",
            "sub": "Reservoir Rock",
            "color": "#4ECDC4",
            "hatch": "...",
            "alpha": 0.7,
        },
        # Right Heat Source Rock (L1) - beyond F4
        {
            "x": x5,
            "y": y0,
            "width": x6 - x5,
            "height": y2 - y0,
            "label": "L1",
            "sub": "Heat Source Rock",
            "color": "#FF6B6B",
            "hatch": "///",
            "alpha": 0.7,
        },
    ]

    # Draw geological blocks
    for block in blocks:
        rect = Rectangle(
            (block["x"], block["y"]),
            block["width"],
            block["height"],
            facecolor=block["color"],
            edgecolor="black",
            linewidth=2.5,
            alpha=block["alpha"],
            hatch=block["hatch"],
            zorder=1,
        )
        ax.add_patch(rect)

        # Add main label
        center_x = block["x"] + block["width"] / 2
        center_y = block["y"] + block["height"] / 2

        ax.text(
            center_x,
            center_y + 0.9,
            block["label"],
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
        )
        ax.text(
            center_x,
            center_y + 1.4,
            block["sub"],
            ha="center",
            va="center",
            fontsize=11,
            fontstyle="italic",
        )

    # Draw fault lines (F1-F4)
    faults = [
        {"x": x0, "label": "F4", "linestyle": "--", "color": "red", "linewidth": 3},
        {"x": x1, "label": "F1", "linestyle": "-", "color": "red", "linewidth": 3},
        {"x": x2, "label": "F2", "linestyle": "-", "color": "red", "linewidth": 3},
        {"x": x4, "label": "F3", "linestyle": "-", "color": "red", "linewidth": 3},
        {"x": x5, "label": "F4", "linestyle": "--", "color": "red", "linewidth": 3},
    ]

    for fault in faults:
        ax.axvline(
            x=fault["x"],
            ymin=y0 / (y2 + 0.5),
            ymax=y2 / (y2 + 0.5),
            linestyle=fault["linestyle"],
            color=fault["color"],
            linewidth=fault["linewidth"],
            zorder=2,
        )
        ax.text(
            fault["x"],
            y2 + 0.6,
            fault["label"],
            ha="center",
            va="bottom",
            fontsize=18,
            fontweight="bold",
            color="red",
        )

        # Add coordinate label for fault position
        ax.text(
            fault["x"],
            -0.8,
            f"x={fault['x']:.1f}",
            ha="center",
            va="top",
            fontsize=10,
            color="red",
            fontweight="bold",
        )

    # Draw Geothermal Reservoir (G) - the target
    g_x = (x2 + x4) / 2  # Centroid of L3 block
    g_y = (y0 + y2) / 2

    # Draw G as a star/burst pattern with glow effect
    # Glow effect
    for i in range(3):
        circle = plt.Circle(
            (g_x, g_y), 0.6 + i * 0.3, color="#FF1493", alpha=0.1, zorder=0
        )
        ax.add_patch(circle)

    # Star shape
    star_points = []
    for i in range(12):
        angle = i * 2 * np.pi / 12
        radius = 0.7 if i % 2 == 0 else 0.35
        star_points.append(
            (
                g_x + radius * np.cos(angle - np.pi / 6),
                g_y + radius * np.sin(angle - np.pi / 6),
            )
        )

    star = Polygon(
        star_points,
        closed=True,
        facecolor="#FF1493",
        edgecolor="black",
        linewidth=2.5,
        alpha=0.9,
        zorder=3,
    )
    ax.add_patch(star)

    # Add G label with coordinate
    ax.text(
        g_x,
        g_y - 1.2,
        "G",
        ha="center",
        va="top",
        fontsize=24,
        fontweight="bold",
        color="#FF1493",
    )

    # Add target coordinate annotation
    ax.text(
        g_x,
        g_y - 1.8,
        f"G({g_x:.1f}, {g_y:.1f})",
        ha="center",
        va="top",
        fontsize=12,
        color="#FF1493",
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="#FF1493", alpha=0.8
        ),
    )

    # Draw coordinate grid
    # Major grid lines
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=1, zorder=0)

    # Add major tick marks
    x_ticks = np.arange(0, 12.1, 1)
    y_ticks = np.arange(0, 8.1, 1)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # Add minor grid lines
    ax.grid(True, alpha=0.15, linestyle=":", linewidth=0.5, which="minor")
    ax.minorticks_on()

    # Highlight the target zone (L3 between F2 and F3)
    target_zone = Rectangle(
        (x2, y0),
        x4 - x2,
        y2 - y0,
        fill=False,
        edgecolor="purple",
        linewidth=4,
        linestyle="-",
        alpha=0.7,
        zorder=2,
    )
    ax.add_patch(target_zone)

    # Add target zone label with coordinates
    ax.text(
        (x2 + x4) / 2,
        y2 + 1.2,
        "TARGET ZONE",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="purple",
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor="white", edgecolor="purple", alpha=0.9
        ),
    )

    # Add coordinate labels for target zone boundaries
    ax.text(
        x2,
        y2 + 1.0,
        f"x₂={x2:.1f}",
        ha="center",
        va="bottom",
        fontsize=10,
        color="purple",
        fontweight="bold",
    )
    ax.text(
        x4,
        y2 + 1.0,
        f"x₄={x4:.1f}",
        ha="center",
        va="bottom",
        fontsize=10,
        color="purple",
        fontweight="bold",
    )
    ax.text(
        x2 - 0.5,
        y0,
        f"y₀={y0:.1f}",
        ha="right",
        va="bottom",
        fontsize=10,
        color="purple",
        fontweight="bold",
    )
    ax.text(
        x2 - 0.5,
        y2,
        f"y₂={y2:.1f}",
        ha="right",
        va="bottom",
        fontsize=10,
        color="purple",
        fontweight="bold",
    )

    # Add equation box for target calculation
    equation_text = f"G = ((x₂+x₄)/2, (y₀+y₂)/2)\nG = (({x2:.1f}+{x4:.1f})/2, ({y0:.1f}+{y2:.1f})/2)\nG = ({g_x:.1f}, {g_y:.1f})"
    ax.text(
        x6 + 1.8,
        (y0 + y2) / 2,
        equation_text,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="lightyellow",
            edgecolor="black",
            alpha=0.9,
        ),
    )

    # Add annotation boxes for key principles with coordinates
    principles = [
        ("Contiguity Axiom:\nL1 | L2 | L3", (x0 + (x1 - x0) / 2, y2 + 2.0), 0.9),
        ("Boundary Axiom:\nFaults at contacts", (x1 + (x2 - x1) / 2, y2 + 2.0), 0.9),
        ("Trap Axiom:\nReservoir + Seal + Faults", (x2 + (x4 - x2) / 2, y2 + 2.0), 0.9),
        (
            "Exclusion Rule:\nNo G in L1 or unsealed L2",
            (x4 + (x5 - x4) / 2, y2 + 2.0),
            0.9,
        ),
    ]

    for text, pos, alpha in principles:
        ax.text(
            pos[0],
            pos[1],
            text,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="lightgray",
                edgecolor="black",
                alpha=alpha,
            ),
        )

    # Add arrows showing fluid flow and heat with coordinate positions
    # Heat from L1
    ax.annotate(
        "Heat Flow ↑",
        xy=((x0 + x1) / 2, y0 + 0.5),
        xytext=((x0 + x1) / 2, y0 - 1.5),
        arrowprops=dict(arrowstyle="->", lw=2.5, color="orange"),
        ha="center",
        va="top",
        fontsize=12,
        color="orange",
        fontweight="bold",
    )

    # Fluid flow through L2
    ax.annotate(
        "Fluid Flow →",
        xy=(x1 + (x2 - x1) / 2, (y0 + y2) / 2),
        xytext=(x1 + (x2 - x1) / 2, (y0 + y2) / 2 - 1.5),
        arrowprops=dict(arrowstyle="->", lw=2.5, color="blue"),
        ha="center",
        va="top",
        fontsize=12,
        color="blue",
        fontweight="bold",
    )

    # Confinement arrows on faults
    ax.annotate(
        "Structural\nConfinement",
        xy=(x2, (y0 + y2) / 2),
        xytext=(x2 - 2.0, (y0 + y2) / 2 + 1.0),
        arrowprops=dict(arrowstyle="<->", lw=2, color="purple"),
        ha="center",
        va="center",
        fontsize=10,
        color="purple",
    )

    # ============================================
    # NORTH ARROW
    # ============================================
    north_x = 10.0
    north_y = 7.5

    # North arrow body (line)
    ax.annotate(
        "",
        xy=(north_x, north_y + 1.0),
        xytext=(north_x, north_y),
        arrowprops=dict(arrowstyle="->", lw=3, color="black", mutation_scale=30),
    )

    # North arrowhead (filled triangle)
    ax.plot(
        north_x,
        north_y + 1.0,
        "^",
        markersize=15,
        color="black",
        markeredgecolor="black",
    )

    # "N" label
    ax.text(
        north_x - 0.12,
        north_y + 1.3,
        "N",
        fontsize=16,
        fontweight="bold",
        color="black",
        ha="left",
        va="center",
    )

    # ============================================
    # SCALE BAR
    # ============================================
    scale_bar_x = 8.5
    scale_bar_y = -0.3

    # Scale bar length in coordinate units (2 units)
    scale_length = 2.0

    # Draw scale bar (horizontal line)
    ax.plot(
        [scale_bar_x, scale_bar_x + scale_length],
        [scale_bar_y, scale_bar_y],
        "k-",
        linewidth=3,
    )

    # End ticks
    ax.plot(
        [scale_bar_x, scale_bar_x],
        [scale_bar_y - 0.15, scale_bar_y + 0.15],
        "k-",
        linewidth=2,
    )
    ax.plot(
        [scale_bar_x + scale_length, scale_bar_x + scale_length],
        [scale_bar_y - 0.15, scale_bar_y + 0.15],
        "k-",
        linewidth=2,
    )

    # Scale bar labels
    ax.text(
        scale_bar_x,
        scale_bar_y - 0.3,
        "0",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        scale_bar_x + scale_length,
        scale_bar_y - 0.3,
        f"{scale_length:.0f} km",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
    )

    # Scale bar label
    ax.text(
        scale_bar_x + scale_length / 2,
        scale_bar_y - 0.9,
        "Scale",
        ha="center",
        va="top",
        fontsize=10,
        fontstyle="italic",
    )

    # Add coordinate axis labels with arrowheads
    # X-axis
    ax.annotate(
        "",
        xy=(11.5, -0.5),
        xytext=(-0.5, -0.5),
        arrowprops=dict(arrowstyle="->", lw=2, color="black"),
    )
    ax.text(11.8, -0.5, "X", fontsize=14, fontweight="bold")

    # Y-axis
    ax.annotate(
        "",
        xy=(-0.5, 7.5),
        xytext=(-0.5, -0.5),
        arrowprops=dict(arrowstyle="->", lw=2, color="black"),
    )
    ax.text(-0.5, 7.8, "Y", fontsize=14, fontweight="bold")

    # Set axis limits and labels
    ax.set_xlim(-1, 12.5)
    ax.set_ylim(-1, 9.5)

    # Add main title
    ax.set_title(
        "ZEYBEK-2 Model: Geometry-Driven Targeting System with Coordinate Framework",
        fontsize=18,
        fontweight="bold",
        pad=25,
    )

    # Add coordinate system labels
    ax.set_xlabel(
        "Horizontal Distance (East) (x) - Cartesian Coordinate System",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel(
        "Vertical Depth (North) (y) - Cartesian Coordinate System",
        fontsize=14,
        fontweight="bold",
    )

    # Add legend
    legend_elements = [
        patches.Patch(facecolor="#FF6B6B", alpha=0.7, label="L1: Heat Source Rock"),
        patches.Patch(facecolor="#4ECDC4", alpha=0.7, label="L2: Reservoir Rock"),
        patches.Patch(facecolor="#FFE66D", alpha=0.7, label="L3: Cap Rock (Seal)"),
        patches.Patch(
            facecolor="#FF1493",
            alpha=0.8,
            label="G: Geothermal Reservoir (Target)",
        ),
        plt.Line2D(
            [0],
            [0],
            color="red",
            linestyle="-",
            linewidth=2,
            label="F1-F3: Bounding Faults",
        ),
        plt.Line2D(
            [0],
            [0],
            color="red",
            linestyle="--",
            linewidth=2,
            label="F4: External Fault",
        ),
        patches.Patch(
            facecolor="none",
            edgecolor="purple",
            linewidth=2,
            label="Target Zone (L3 between F2-F3)",
        ),
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(1.35, 1.0),
        fontsize=11,
        framealpha=0.95,
        borderpad=1.5,
    )

    # Add coordinate value table
    coord_table = [
        ["Element", "x-coordinate", "y-coordinate"],
        ["F4", f"{x0:.1f}", f"{y0:.1f} - {y2:.1f}"],
        ["F1", f"{x1:.1f}", f"{y0:.1f} - {y2:.1f}"],
        ["F2", f"{x2:.1f}", f"{y0:.1f} - {y2:.1f}"],
        ["F3", f"{x4:.1f}", f"{y0:.1f} - {y2:.1f}"],
        ["L3 (Cap Rock)", f"{x2:.1f} - {x4:.1f}", f"{y0:.1f} - {y2:.1f}"],
        ["G (Target)", f"{g_x:.1f}", f"{g_y:.1f}"],
    ]

    # Create table
    table = ax.table(
        cellText=coord_table,
        loc="lower right",
        cellLoc="center",
        colWidths=[0.15, 0.15, 0.2],
        bbox=[1.02, 0.02, 0.26, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Style the table
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#40466e")
            cell.set_text_props(weight="bold", color="white")
        else:
            cell.set_facecolor("#f0f0f0")

    plt.tight_layout()
    return fig, ax


# Main execution
if __name__ == "__main__":
    # Create 2D figure with coordinates
    fig1, ax1 = draw_zeybek2_with_coordinates()
    plt.savefig("zeybek2_model_with_coordinates.png", dpi=300, bbox_inches="tight")
    plt.savefig("zeybek2_model_with_coordinates.pdf", bbox_inches="tight")
    plt.show()

XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Patch
import matplotlib

# Use non-interactive backend if needed
matplotlib.use("Agg")


def create_geothermal_diagram():
    """Create a simplified geothermal system diagram with proper legend and labels"""

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Define blocks with their properties - REVISED POSITIONS
    blocks = [
        # Left L1 (Heat Source) - extending out and completely below L2
        {
            "x1": -7.5,
            "x2": -5.5,
            "y1": -2.0,
            "y2": 2.0,
            "z1": -4.0,
            "z2": -1.5,
            "color": "#ff6b35",
            "label": "L1",
            "alpha": 0.85,
        },
        # Left L2 (Reservoir) - completely above L1 and below L3
        {
            "x1": -5.5,
            "x2": -1.5,
            "y1": -1.8,
            "y2": 1.8,
            "z1": -1.5,
            "z2": 0.0,
            "color": "#4ecdc4",
            "label": "L2",
            "alpha": 0.85,
        },
        # L3 (Cap Rock) - at the top center
        {
            "x1": -1.5,
            "x2": 1.5,
            "y1": -1.8,
            "y2": 1.8,
            "z1": 0.0,
            "z2": 1.5,
            "color": "#ffe66d",
            "label": "L3",
            "alpha": 0.85,
        },
        # Right L2 (Reservoir) - completely above L1 and below L3
        {
            "x1": 1.5,
            "x2": 5.5,
            "y1": -1.8,
            "y2": 1.8,
            "z1": -1.5,
            "z2": 0.0,
            "color": "#4ecdc4",
            "label": "L2",
            "alpha": 0.85,
        },
        # Right L1 (Heat Source) - extending out and completely below L2
        {
            "x1": 5.5,
            "x2": 7.5,
            "y1": -2.0,
            "y2": 2.0,
            "z1": -4.0,
            "z2": -1.5,
            "color": "#ff6b35",
            "label": "L1",
            "alpha": 0.85,
        },
    ]

    def add_box(ax, x1, x2, y1, y2, z1, z2, color, label, alpha):
        """Add a 3D box to the plot"""
        vertices = np.array(
            [
                [x1, y1, z1],
                [x2, y1, z1],
                [x2, y2, z1],
                [x1, y2, z1],
                [x1, y1, z2],
                [x2, y1, z2],
                [x2, y2, z2],
                [x1, y2, z2],
            ]
        )
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[0], vertices[3], vertices[7], vertices[4]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
        ]
        poly = Poly3DCollection(faces, alpha=alpha, edgecolor="black", linewidth=1.0)
        poly.set_facecolor(color)
        ax.add_collection3d(poly)

        # REMOVED: Label text on each box
        # No text added to the boxes

    # Add all blocks
    for block in blocks:
        add_box(
            ax,
            block["x1"],
            block["x2"],
            block["y1"],
            block["y2"],
            block["z1"],
            block["z2"],
            block["color"],
            block["label"],
            block["alpha"],
        )

    # Draw fault lines with prominent labels F1, F2, F3, F4
    fault_positions = [-5.5, -1.5, 1.5, 5.5]
    fault_labels = ["F1", "F2", "F3", "F4"]

    for i, (x, f_label) in enumerate(zip(fault_positions, fault_labels)):
        # Draw vertical fault lines - extending through all layers
        ax.plot([x, x], [-2.5, 2.5], [-4.5, 2.0], "k--", linewidth=3, alpha=0.9)

        # REMOVED: Fault labels at top and bottom
        # No text added for faults

    # REMOVED: All structural labels (Graben, Horst, G, depth labels, etc.)

    # REMOVED: Geothermal target G and glow effects

    # REMOVED: Temperature and fluid flow indicators

    # REMOVED: Depth labels on the side

    # Create legend with the specified labels
    legend_elements = [
        Patch(
            facecolor="#ff6b35",
            alpha=0.85,
            edgecolor="black",
            label="L1: Heat Source Rock",
        ),
        Patch(
            facecolor="#4ecdc4",
            alpha=0.85,
            edgecolor="black",
            label="L2: Reservoir Rock",
        ),
        Patch(
            facecolor="#ffe66d",
            alpha=0.85,
            edgecolor="black",
            label="L3: Cap Rock (Seal)",
        ),
        Patch(
            facecolor="none",
            edgecolor="black",
            linestyle="--",
            linewidth=3,
            label="F1-F4: Fault Lines",
        ),
    ]

    # Add legend to the figure
    legend = ax.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=12,
        framealpha=0.95,
        edgecolor="black",
        bbox_to_anchor=(0.02, 0.98),
        ncol=1,
    )
    legend.get_frame().set_facecolor("white")

    # Add a title with subtitle
    ax.set_xlabel("X Distance (km) (East)", fontsize=13, labelpad=12, weight="bold")
    ax.set_ylabel("Y Distance (km) (North)", fontsize=13, labelpad=12, weight="bold")
    ax.set_zlabel("Depth (km)", fontsize=13, labelpad=12, weight="bold")

    plt.suptitle(
        "3D Horst-Graben Geothermal System Model", fontsize=18, weight="bold", y=0.98
    )
    ax.set_title(
        "ZEYBEK-2 Geothermal Prospect\nStructural controls on geothermal resources",
        fontsize=13,
        pad=20,
        style="italic",
    )

    # View settings for optimal visualization
    ax.view_init(elev=55, azim=-77)
    ax.invert_zaxis()
    ax.set_xlim([-9.0, 9.0])
    ax.set_ylim([-3.5, 3.5])
    ax.set_zlim([-3.5, 3.5])

    # Enhanced grid
    ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)

    # Set axis ticks - Depth axis will be clearly visible
    ax.set_xticks([-7, -5, -3, -1, 0, 1, 3, 5, 7])
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_zticks([-4, -3, -2, -1, 0, 1])

    # Make depth axis labels more prominent
    ax.zaxis.label.set_size(14)
    ax.zaxis.label.set_weight("bold")

    plt.tight_layout()
    return fig


# Create and save the figure
print("Generating geothermal system diagram with legend only...")
fig = create_geothermal_diagram()
fig.savefig("zeybek_geothermal_3d_legend_only.png", dpi=300, bbox_inches="tight")
print("✓ Diagram saved as 'zeybek_geothermal_3d_legend_only.png'")
print("\n✓ All figure text removed:")
print("  - No L1, L2, L3 labels on blocks")
print("  - No F1, F2, F3, F4 labels on fault lines")
print("  - No Graben/Horst labels")
print("  - No G target label")
print("  - No depth annotations or temperature indicators")
print("  - Legend retained with all descriptions")
print("  - Depth axis (km) clearly visible with tick marks")

# If you want to try displaying it interactively
try:
    plt.show()
except:
    print("\nInteractive display not available. Image has been saved as PNG file.")

XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
