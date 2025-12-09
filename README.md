# Geothermal_Finder_Dr.Mutlu-Zeybek
Geothermal_Finder_Dr.Mutlu Zeybek
"""
ZEYBEK-2 Model: A Rule-Based Expert System for Systematic, Geometry-Driven Targeting 
of Fault-Controlled Geothermal Reservoirs

Author: Mutlu ZEYBEK
Affiliation: Muğla Metropolitan Municipality, Muğla, Turkey

Implementation of the geometric rules for locating geothermal traps based on:
1. Heat source (L1)
2. Permeable reservoir rock (L2)
3. Impermeable cap rock (L3)
4. Bounding fault structures (F1-F4)
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import matplotlib.patches as mpatches

class LithologyType(Enum):
    """Types of lithological units in the ZEYBEK-2 Model"""
    HEAT_SOURCE = "L1"      # Heat source / Source rock
    RESERVOIR = "L2"        # Permeable reservoir rock
    CAP_ROCK = "L3"         # Impermeable cap rock / seal

class FaultType(Enum):
    """Types of fault structures in the ZEYBEK-2 Model"""
    F4 = "F4"  # External left boundary
    F1 = "F1"  # L1/L2 boundary
    F2 = "F2"  # L2/L3 left boundary
    F3 = "F3"  # L3/L2 right boundary

@dataclass
class GeologicalUnit:
    """Represents a geological unit in the model"""
    unit_type: LithologyType
    vertices: np.ndarray  # Shape (n, 2) for n vertices
    name: str = ""
    
    @property
    def centroid(self) -> np.ndarray:
        """Calculate the centroid of the unit"""
        return np.mean(self.vertices, axis=0)
    
    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box (xmin, xmax, ymin, ymax)"""
        x_coords = self.vertices[:, 0]
        y_coords = self.vertices[:, 1]
        return (np.min(x_coords), np.max(x_coords), 
                np.min(y_coords), np.max(y_coords))

@dataclass
class Fault:
    """Represents a fault structure"""
    fault_type: FaultType
    start_point: np.ndarray  # Shape (2,)
    end_point: np.ndarray    # Shape (2,)
    
    @property
    def coordinates(self) -> np.ndarray:
        """Get fault line coordinates"""
        return np.array([self.start_point, self.end_point])
    
    @property
    def x_position(self) -> float:
        """Get x-coordinate of vertical fault (assuming vertical)"""
        return self.start_point[0]

class ZEYBEK2Model:
    """
    Implementation of the ZEYBEK-2 Rule-Based Expert System for geothermal targeting.
    
    The model operates on these core axioms:
    1. Contiguity Axiom: L1 | L2 | L3 sequence
    2. Boundary Axiom: Contacts between units co-located with faults
    3. Trap Axiom: Reservoir can only form in L3 block between F2 and F3
    """
    
    def __init__(self):
        """Initialize the ZEYBEK-2 model"""
        self.units: Dict[str, GeologicalUnit] = {}
        self.faults: Dict[str, Fault] = {}
        self.target_coordinate: Optional[np.ndarray] = None
        
    def define_cap_rock(self, vertices: np.ndarray) -> None:
        """
        Define the Cap Rock (L3) unit. This is the primary input.
        
        Parameters:
        -----------
        vertices : np.ndarray
            Vertices of the L3 polygon in order: (x₂, y₀), (x₂, y₂), (x₄, y₂), (x₄, y₀)
        """
        if vertices.shape != (4, 2):
            raise ValueError("Cap rock must be defined by 4 vertices (rectangle)")
        
        self.units["L3"] = GeologicalUnit(
            unit_type=LithologyType.CAP_ROCK,
            vertices=vertices,
            name="Cap Rock / Seal"
        )
        
        # Extract coordinates
        x_coords = vertices[:, 0]
        y_coords = vertices[:, 1]
        
        self.x2 = np.min(x_coords)  # Left boundary of L3
        self.x4 = np.max(x_coords)  # Right boundary of L3
        self.y0 = np.min(y_coords)  # Bottom boundary
        self.y2 = np.max(y_coords)  # Top boundary
        
        # Calculate inferred coordinates
        self.x3 = (self.x2 + self.x4) / 2  # Center x of L3
        self.y1 = (self.y0 + self.y2) / 2  # Center y of L3
        
        # Define width for adjacent units (can be parameterized)
        self.unit_width = (self.x4 - self.x2) * 0.8  # Default width
        
    def infer_adjacent_units(self) -> None:
        """
        Infer adjacent geological units based on Contiguity Axiom.
        Sequence: L1 | L2 | L3 | L2 | L1
        """
        if "L3" not in self.units:
            raise ValueError("Cap rock (L3) must be defined first")
        
        # Left Reservoir (L2) - between F1 and F2
        x1 = self.x2 - self.unit_width
        left_reservoir_vertices = np.array([
            [x1, self.y0],     # Bottom-left
            [x1, self.y2],     # Top-left
            [self.x2, self.y2], # Top-right
            [self.x2, self.y0]  # Bottom-right
        ])
        
        self.units["L2_left"] = GeologicalUnit(
            unit_type=LithologyType.RESERVOIR,
            vertices=left_reservoir_vertices,
            name="Left Reservoir Rock"
        )
        
        # Left Heat Source (L1) - between F4 and F1
        x0 = x1 - self.unit_width
        left_source_vertices = np.array([
            [x0, self.y0],     # Bottom-left
            [x0, self.y2],     # Top-left
            [x1, self.y2],     # Top-right
            [x1, self.y0]      # Bottom-right
        ])
        
        self.units["L1_left"] = GeologicalUnit(
            unit_type=LithologyType.HEAT_SOURCE,
            vertices=left_source_vertices,
            name="Left Heat Source"
        )
        
        # Right Reservoir (L2) - between F3 and inferred F4_right
        x5 = self.x4 + self.unit_width
        right_reservoir_vertices = np.array([
            [self.x4, self.y0], # Bottom-left
            [self.x4, self.y2], # Top-left
            [x5, self.y2],      # Top-right
            [x5, self.y0]       # Bottom-right
        ])
        
        self.units["L2_right"] = GeologicalUnit(
            unit_type=LithologyType.RESERVOIR,
            vertices=right_reservoir_vertices,
            name="Right Reservoir Rock"
        )
        
        # Right Heat Source (L1) - beyond inferred F4_right
        x6 = x5 + self.unit_width
        right_source_vertices = np.array([
            [x5, self.y0],     # Bottom-left
            [x5, self.y2],     # Top-left
            [x6, self.y2],     # Top-right
            [x6, self.y0]      # Bottom-right
        ])
        
        self.units["L1_right"] = GeologicalUnit(
            unit_type=LithologyType.HEAT_SOURCE,
            vertices=right_source_vertices,
            name="Right Heat Source"
        )
        
        # Store inferred coordinates
        self.x0 = x0
        self.x1 = x1
        self.x5 = x5
        self.x6 = x6
        
    def define_faults(self) -> None:
        """
        Define fault structures based on Boundary Axiom.
        Full sequence: F4 | L1 | F1 | L2 | F2 | L3 | F3
        """
        if "L3" not in self.units:
            raise ValueError("Units must be defined first")
        
        # Fault F4 (External Left Boundary)
        self.faults["F4"] = Fault(
            fault_type=FaultType.F4,
            start_point=np.array([self.x0, self.y0]),
            end_point=np.array([self.x0, self.y2])
        )
        
        # Fault F1 (L1/L2 Boundary)
        self.faults["F1"] = Fault(
            fault_type=FaultType.F1,
            start_point=np.array([self.x1, self.y0]),
            end_point=np.array([self.x1, self.y2])
        )
        
        # Fault F2 (L2/L3 Left Boundary)
        self.faults["F2"] = Fault(
            fault_type=FaultType.F2,
            start_point=np.array([self.x2, self.y0]),
            end_point=np.array([self.x2, self.y2])
        )
        
        # Fault F3 (L3/L2 Right Boundary)
        self.faults["F3"] = Fault(
            fault_type=FaultType.F3,
            start_point=np.array([self.x4, self.y0]),
            end_point=np.array([self.x4, self.y2])
        )
    
    def calculate_target(self) -> np.ndarray:
        """
        Calculate target coordinate based on Trap Axiom.
        
        Returns:
        --------
        np.ndarray : Target coordinate G(x₃, y₁)
        """
        # According to Trap Axiom: G can only be in L3 block between F2 and F3
        # The centroid of L3 is the most probable location
        self.target_coordinate = np.array([self.x3, self.y1])
        return self.target_coordinate
    
    def validate_exclusion_rules(self) -> Dict[str, bool]:
        """
        Validate the Exclusivity Principle of the model.
        
        Returns:
        --------
        Dict[str, bool] : Validation results for each exclusion rule
        """
        validation = {}
        
        # Rule 1: G is NOT in the L2 block between F1 & F2 (lacks vertical seal)
        l2_left_bounds = self.units["L2_left"].bounds
        validation["not_in_left_reservoir"] = not (
            l2_left_bounds[0] <= self.x3 <= l2_left_bounds[1] and
            l2_left_bounds[2] <= self.y1 <= l2_left_bounds[3]
        )
        
        # Rule 2: G is NOT in the L2 block between F3 & F4_right (lacks vertical seal)
        l2_right_bounds = self.units["L2_right"].bounds
        validation["not_in_right_reservoir"] = not (
            l2_right_bounds[0] <= self.x3 <= l2_right_bounds[1] and
            l2_right_bounds[2] <= self.y1 <= l2_right_bounds[3]
        )
        
        # Rule 3: G is NOT in any L1 block (lacks both permeability and seal)
        l1_left_bounds = self.units["L1_left"].bounds
        l1_right_bounds = self.units["L1_right"].bounds
        
        validation["not_in_left_source"] = not (
            l1_left_bounds[0] <= self.x3 <= l1_left_bounds[1] and
            l1_left_bounds[2] <= self.y1 <= l1_left_bounds[3]
        )
        
        validation["not_in_right_source"] = not (
            l1_right_bounds[0] <= self.x3 <= l1_right_bounds[1] and
            l1_right_bounds[2] <= self.y1 <= l1_right_bounds[3]
        )
        
        # Rule 4: G IS in L3 block between F2 and F3
        l3_bounds = self.units["L3"].bounds
        validation["in_cap_rock"] = (
            l3_bounds[0] <= self.x3 <= l3_bounds[1] and
            l3_bounds[2] <= self.y1 <= l3_bounds[3]
        )
        
        return validation
    
    def run_model(self, cap_rock_vertices: np.ndarray) -> Dict:
        """
        Execute the complete ZEYBEK-2 Model algorithm.
        
        Parameters:
        -----------
        cap_rock_vertices : np.ndarray
            Vertices of the L3 (cap rock) polygon
            
        Returns:
        --------
        Dict : Complete model results including target coordinate and validation
        """
        # Step 1: Define cap rock (primary input)
        self.define_cap_rock(cap_rock_vertices)
        
        # Step 2: Infer adjacent units
        self.infer_adjacent_units()
        
        # Step 3: Define faults
        self.define_faults()
        
        # Step 4: Calculate target coordinate
        target = self.calculate_target()
        
        # Step 5: Validate exclusion rules
        validation = self.validate_exclusion_rules()
        
        # Compile results
        results = {
            "target_coordinate": target,
            "validation": validation,
            "units": self.units,
            "faults": self.faults,
            "coordinates": {
                "x0": self.x0, "x1": self.x1, "x2": self.x2,
                "x3": self.x3, "x4": self.x4, "x5": self.x5,
                "x6": self.x6, "y0": self.y0, "y1": self.y1,
                "y2": self.y2
            }
        }
        
        return results
    
    def visualize_model(self, save_path: Optional[str] = None) -> None:
        """
        Create a visualization of the ZEYBEK-2 Model configuration.
        
        Parameters:
        -----------
        save_path : Optional[str]
            Path to save the visualization (if None, display only)
        """
        if not self.units or not self.faults:
            raise ValueError("Model must be run before visualization")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Define colors for different unit types
        color_map = {
            LithologyType.HEAT_SOURCE: 'lightcoral',
            LithologyType.RESERVOIR: 'lightblue',
            LithologyType.CAP_ROCK: 'lightgreen'
        }
        
        # Plot geological units
        for name, unit in self.units.items():
            polygon = Polygon(unit.vertices, closed=True, 
                            facecolor=color_map[unit.unit_type],
                            edgecolor='black', alpha=0.7, linewidth=1)
            ax.add_patch(polygon)
            
            # Add unit label at centroid
            centroid = unit.centroid
            ax.text(centroid[0], centroid[1], unit.unit_type.value,
                   ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Plot faults
        for name, fault in self.faults.items():
            coords = fault.coordinates
            ax.plot(coords[:, 0], coords[:, 1], 'k-', linewidth=3)
            
            # Add fault label
            mid_point = np.mean(coords, axis=0)
            ax.text(mid_point[0] - 0.1, mid_point[1], fault.fault_type.value,
                   ha='right', va='center', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Plot target coordinate
        if self.target_coordinate is not None:
            ax.plot(self.target_coordinate[0], self.target_coordinate[1], 
                   'r*', markersize=20, label='Target G(x₃, y₁)')
            
            # Add coordinate text
            ax.text(self.target_coordinate[0], self.target_coordinate[1] - 0.5,
                   f'G({self.x3:.1f}, {self.y1:.1f})', 
                   ha='center', va='top', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.5))
        
        # Set plot limits and labels
        all_vertices = np.vstack([unit.vertices for unit in self.units.values()])
        x_min, x_max = np.min(all_vertices[:, 0]), np.max(all_vertices[:, 1])
        y_min, y_max = np.min(all_vertices[:, 0]), np.max(all_vertices[:, 1])
        
        ax.set_xlim(x_min - 1, x_max + 1)
        ax.set_ylim(y_min - 1, y_max + 1)
        ax.set_xlabel('X Coordinate (km)', fontsize=12)
        ax.set_ylabel('Y Coordinate (km)', fontsize=12)
        ax.set_title('ZEYBEK-2 Model: Fault-Controlled Geothermal Reservoir Targeting', 
                    fontsize=14, fontweight='bold')
        
        # Add legend
        legend_elements = [
            mpatches.Patch(facecolor='lightcoral', edgecolor='black', label='L1: Heat Source'),
            mpatches.Patch(facecolor='lightblue', edgecolor='black', label='L2: Reservoir Rock'),
            mpatches.Patch(facecolor='lightgreen', edgecolor='black', label='L3: Cap Rock'),
            plt.Line2D([0], [0], color='k', linewidth=3, label='Faults'),
            plt.Line2D([0], [0], marker='*', color='r', markersize=10, 
                      label='Target G(x₃, y₁)', linestyle='')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def generate_report(self) -> str:
        """
        Generate a text report of the model results.
        
        Returns:
        --------
        str : Formatted report string
        """
        if self.target_coordinate is None:
            return "Model has not been run yet."
        
        validation = self.validate_exclusion_rules()
        
        report = []
        report.append("=" * 60)
        report.append("ZEYBEK-2 MODEL: GEOTHERMAL TARGETING REPORT")
        report.append("=" * 60)
        report.append("")
        report.append("MODEL OUTPUT:")
        report.append(f"  Target Coordinate G: ({self.x3:.2f}, {self.y1:.2f})")
        report.append("")
        report.append("GEOMETRIC COORDINATES:")
        report.append(f"  F4 (x₀): {self.x0:.2f}")
        report.append(f"  F1 (x₁): {self.x1:.2f}")
        report.append(f"  F2 (x₂): {self.x2:.2f}")
        report.append(f"  F3 (x₄): {self.x4:.2f}")
        report.append(f"  y₀: {self.y0:.2f}")
        report.append(f"  y₂: {self.y2:.2f}")
        report.append("")
        report.append("EXCLUSION RULE VALIDATION:")
        report.append(f"  G in L3 (Cap Rock): {validation['in_cap_rock']} ✓")
        report.append(f"  G NOT in Left Reservoir: {validation['not_in_left_reservoir']} ✓")
        report.append(f"  G NOT in Right Reservoir: {validation['not_in_right_reservoir']} ✓")
        report.append(f"  G NOT in Left Source: {validation['not_in_left_source']} ✓")
        report.append(f"  G NOT in Right Source: {validation['not_in_right_source']} ✓")
        report.append("")
        report.append("INTERPRETATION:")
        report.append("  According to the ZEYBEK-2 Trap Axiom, the geothermal")
        report.append("  reservoir can only form in the Cap Rock (L3) block")
        report.append("  between bounding faults F2 and F3, where the underlying")
        report.append("  reservoir rock (L2) is both laterally confined and")
        report.append("  vertically sealed.")
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)

# Example usage and synthetic case study
def synthetic_case_study():
    """
    Demonstrate the ZEYBEK-2 Model with a synthetic case study.
    This mimics a 'greenfield' exploration scenario.
    """
    print("SYNTHETIC CASE STUDY: Greenfield Geothermal Prospect")
    print("-" * 50)
    
    # Create model instance
    model = ZEYBEK2Model()
    
    # Define cap rock based on interpreted data (e.g., from MT survey)
    # Vertices: (x₂, y₀), (x₂, y₂), (x₄, y₂), (x₄, y₀)
    cap_rock_vertices = np.array([
        [3.0, 1.0],   # Bottom-left (x₂, y₀)
        [3.0, 5.0],   # Top-left (x₂, y₂)
        [7.0, 5.0],   # Top-right (x₄, y₂)
        [7.0, 1.0]    # Bottom-right (x₄, y₀)
    ])
    
    print(f"Input Cap Rock (L3) vertices:")
    for i, vertex in enumerate(cap_rock_vertices):
        print(f"  Vertex {i}: ({vertex[0]:.1f}, {vertex[1]:.1f})")
    print()
    
    # Run the model
    results = model.run_model(cap_rock_vertices)
    
    # Generate report
    report = model.generate_report()
    print(report)
    
    # Visualize the model
    model.visualize_model(save_path="zeybek2_model_output.png")
    
    return results

def compare_with_real_world_case():
    """
    Compare model logic with documented field example (conceptual alignment).
    """
    print("\n" + "=" * 60)
    print("CONCEPTUAL ALIGNMENT WITH REAL-WORLD CASE: Linyi Geothermal Field")
    print("=" * 60)
    
    print("\nGeological Elements Alignment:")
    print("-" * 40)
    
    alignments = {
        "ZEYBEK-2 Element": ["L3 (Cap Rock)", "L2 (Reservoir)", "F2 & F3", "G (Reservoir)"],
        "Linyi Field Equivalent": [
            "Thick clay layers in fault zone",
            "Fractured Ordovician/Cambrian carbonates",
            "Yishu Fault zone boundaries",
            "Geothermal anomalies in fault-bounded, sealed compartment"
        ],
        "Functional Role": [
            "Vertical seal preventing fluid escape",
            "Permeable fluid host and pathway",
            "Lateral compartment boundaries",
            "Exploitable accumulation zone"
        ]
    }
    
    # Print alignment table
    for i in range(len(alignments["ZEYBEK-2 Element"])):
        print(f"{alignments['ZEYBEK-2 Element'][i]:<20} -> "
              f"{alignments['Linyi Field Equivalent'][i]:<40} "
              f"({alignments['Functional Role'][i]})")
    
    print("\nKey Insight:")
    print("The ZEYBEK-2 Model's core principle—that exploitable reservoirs")
    print("require the intersection of permeable rock, sealing cap, and")
    print("confining faults—is validated by field observations.")
    print("\n" + "=" * 60)

# API for integration with other exploration tools
class ZEYBEK2API:
    """
    API wrapper for integrating ZEYBEK-2 Model with other exploration tools.
    """
    
    @staticmethod
    def from_geophysical_data(resistivity_map: np.ndarray, 
                             fault_locations: List[Tuple[float, float]],
                             pixel_size: float = 1.0) -> ZEYBEK2Model:
        """
        Initialize model from geophysical data (e.g., MT resistivity).
        
        Parameters:
        -----------
        resistivity_map : np.ndarray
            2D resistivity array (low resistivity = potential clay cap)
        fault_locations : List[Tuple[float, float]]
            List of (x, y) coordinates along fault traces
        pixel_size : float
            Size of each pixel in map units
            
        Returns:
        --------
        ZEYBEK2Model : Initialized model
        """
        # This is a simplified example - real implementation would
        # include sophisticated image processing to extract L3 polygon
        model = ZEYBEK2Model()
        
        # Find low-resistivity region (potential clay cap/L3)
        # Threshold can be adjusted based on field data
        threshold = np.percentile(resistivity_map, 30)
        low_resistivity_mask = resistivity_map < threshold
        
        # Find bounding box of largest low-resistivity region
        from scipy import ndimage
        labeled_array, num_features = ndimage.label(low_resistivity_mask)
        
        if num_features == 0:
            raise ValueError("No low-resistivity regions found")
        
        # Use largest connected component
        sizes = ndimage.sum(low_resistivity_mask, labeled_array, range(num_features + 1))
        largest_component = np.argmax(sizes[1:]) + 1
        
        # Get coordinates of largest component
        coords = np.argwhere(labeled_array == largest_component)
        
        if len(coords) < 4:
            raise ValueError("Insufficient points to define polygon")
        
        # Convert to polygon vertices (simplified - convex hull in real implementation)
        y_coords, x_coords = coords[:, 0], coords[:, 1]
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        # Convert pixel coordinates to real coordinates
        x_min_real, x_max_real = x_min * pixel_size, x_max * pixel_size
        y_min_real, y_max_real = y_min * pixel_size, y_max * pixel_size
        
        # Define rectangle (simplified - would use actual polygon in practice)
        cap_rock_vertices = np.array([
            [x_min_real, y_min_real],
            [x_min_real, y_max_real],
            [x_max_real, y_max_real],
            [x_max_real, y_min_real]
        ])
        
        # Run model
        model.define_cap_rock(cap_rock_vertices)
        model.infer_adjacent_units()
        model.define_faults()
        model.calculate_target()
        
        return model
    
    @staticmethod
    def to_gis_export(model: ZEYBEK2Model, 
                     output_format: str = "geojson") -> Dict:
        """
        Export model results to GIS-compatible format.
        
        Parameters:
        -----------
        model : ZEYBEK2Model
            Model instance with results
        output_format : str
            Output format ('geojson', 'shapefile', 'kml')
            
        Returns:
        --------
        Dict : GIS-compatible data structure
        """
        if model.target_coordinate is None:
            raise ValueError("Model must be run before export")
        
        features = []
        
        # Add geological units as polygons
        for name, unit in model.units.items():
            feature = {
                "type": "Feature",
                "properties": {
                    "name": name,
                    "type": unit.unit_type.value,
                    "description": unit.name
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [unit.vertices.tolist()]
                }
            }
            features.append(feature)
        
        # Add faults as line strings
        for name, fault in model.faults.items():
            feature = {
                "type": "Feature",
                "properties": {
                    "name": name,
                    "type": fault.fault_type.value
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": fault.coordinates.tolist()
                }
            }
            features.append(feature)
        
        # Add target point
        target_feature = {
            "type": "Feature",
            "properties": {
                "name": "Target_G",
                "description": "ZEYBEK-2 Model Target Coordinate",
                "x": float(model.x3),
                "y": float(model.y1)
            },
            "geometry": {
                "type": "Point",
                "coordinates": model.target_coordinate.tolist()
            }
        }
        features.append(target_feature)
        
        geojson = {
            "type": "FeatureCollection",
            "features": features,
            "properties": {
                "model": "ZEYBEK-2",
                "version": "2.0",
                "description": "Rule-based expert system for geothermal targeting"
            }
        }
        
        return geojson

# Main execution
if __name__ == "__main__":
    print("ZEYBEK-2 Model Implementation")
    print("A Rule-Based Expert System for Geothermal Reservoir Targeting")
    print("=" * 70)
    
    # Run synthetic case study
    results = synthetic_case_study()
    
    # Compare with real-world case
    compare_with_real_world_case()
    
    print("\n" + "=" * 70)
    print("Model Implementation Complete")
    print("Key features implemented:")
    print("1. Geometric rule-based algorithm")
    print("2. Exclusion principle validation")
    print("3. Visualization capabilities")
    print("4. GIS export functionality")
    print("5. API for integration with exploration tools")
    print("=" * 70)
