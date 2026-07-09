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


def draw_zeybek2_3d_coordinate_system():
    """
    Extended version showing 3D perspective with coordinate axes
    """
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Define coordinates in 3D (x, y, z)
    x0, x1, x2, x3, x4, x5, x6 = 0, 2, 4, 5.5, 7, 9, 11
    y0, y1, y2 = 0, 3.5, 7
    z_min, z_max = 0, 2  # Third dimension (thickness)

    # Create blocks as 3D boxes
    blocks_3d = [
        {
            "x": x0,
            "y": y0,
            "z": z_min,
            "dx": x1 - x0,
            "dy": y2 - y0,
            "dz": z_max - z_min,
            "color": "#FF6B6B",
            "label": "L1",
            "alpha": 0.6,
        },
        {
            "x": x1,
            "y": y0,
            "z": z_min,
            "dx": x2 - x1,
            "dy": y2 - y0,
            "dz": z_max - z_min,
            "color": "#4ECDC4",
            "label": "L2",
            "alpha": 0.6,
        },
        {
            "x": x2,
            "y": y0,
            "z": z_min,
            "dx": x4 - x2,
            "dy": y2 - y0,
            "dz": z_max - z_min,
            "color": "#FFE66D",
            "label": "L3",
            "alpha": 0.6,
        },
        {
            "x": x4,
            "y": y0,
            "z": z_min,
            "dx": x5 - x4,
            "dy": y2 - y0,
            "dz": z_max - z_min,
            "color": "#4ECDC4",
            "label": "L2",
            "alpha": 0.6,
        },
        {
            "x": x5,
            "y": y0,
            "z": z_min,
            "dx": x6 - x5,
            "dy": y2 - y0,
            "dz": z_max - z_min,
            "color": "#FF6B6B",
            "label": "L1",
            "alpha": 0.6,
        },
    ]

    # Draw 3D boxes
    for block in blocks_3d:
        ax.bar3d(
            block["x"],
            block["y"],
            block["z"],
            block["dx"],
            block["dy"],
            block["dz"],
            color=block["color"],
            alpha=block["alpha"],
            edgecolor="black",
        )

    # Draw faults as planes
    for x_pos, label in [(x0, "F4"), (x1, "F1"), (x2, "F2"), (x4, "F3"), (x5, "F4")]:
        # Create fault plane
        y_vals = np.linspace(y0, y2, 10)
        z_vals = np.linspace(z_min, z_max, 10)
        Y, Z = np.meshgrid(y_vals, z_vals)
        X = np.full_like(Y, x_pos)
        ax.plot_surface(X, Y, Z, color="red", alpha=0.3, edgecolor="none")

        # Add fault label
        ax.text(
            x_pos,
            y2 + 0.3,
            (z_max + z_min) / 2,
            label,
            color="red",
            fontsize=14,
            fontweight="bold",
        )

    # Draw target G in 3D
    g_x = (x2 + x4) / 2
    g_y = (y0 + y2) / 2
    g_z = (z_min + z_max) / 2

    # Draw sphere for G
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    sphere_x = g_x + 0.4 * np.outer(np.cos(u), np.sin(v))
    sphere_y = g_y + 0.4 * np.outer(np.sin(u), np.sin(v))
    sphere_z = g_z + 0.4 * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(sphere_x, sphere_y, sphere_z, color="#FF1493", alpha=0.8)

    ax.text(
        g_x,
        g_y,
        g_z + 0.6,
        "G",
        color="#FF1493",
        fontsize=20,
        fontweight="bold",
        ha="center",
    )

    # Coordinate axes
    ax.set_xlabel("X", fontsize=14, fontweight="bold")
    ax.set_ylabel("Y", fontsize=14, fontweight="bold")
    ax.set_zlabel("Z (Thickness)", fontsize=14, fontweight="bold")

    ax.set_title(
        "ZEYBEK-2 Model: 3D Coordinate Representation", fontsize=18, fontweight="bold"
    )

    # Set viewing angle
    ax.view_init(elev=20, azim=-60)

    plt.tight_layout()
    return fig, ax


# Main execution
if __name__ == "__main__":
    # Create 2D figure with coordinates
    fig1, ax1 = draw_zeybek2_with_coordinates()
    plt.savefig("zeybek2_model_with_coordinates.png", dpi=300, bbox_inches="tight")
    plt.savefig("zeybek2_model_with_coordinates.pdf", bbox_inches="tight")
    plt.show()

    # Create 3D version
    fig2, ax2 = draw_zeybek2_3d_coordinate_system()
    plt.savefig("zeybek2_model_3d_coordinates.png", dpi=300, bbox_inches="tight")
    plt.savefig("zeybek2_model_3d_coordinates.pdf", bbox_inches="tight")
    plt.show()

XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
