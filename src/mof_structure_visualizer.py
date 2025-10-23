"""
MOF Structure Visualizer
Generates multi-page PDF visualization of MOF structures from CIF files
Author: zhangshd
Date: 2025-08-26
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional, Tuple
import warnings
import math

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
from matplotlib.patches import Circle
from ase.io import read
from ase.visualize.plot import plot_atoms
from ase.data import covalent_radii
from ase.data.colors import jmol_colors
import numpy as np

warnings.filterwarnings('ignore')


class MOFStructureVisualizer:
    """
    Visualizes MOF structures from CIF files and generates multi-page PDF output.
    
    Each page displays structures in a 3x5 grid (3 columns, 5 rows).
    Empty subplots are added to maintain consistent page layout.
    """
    
    def __init__(self, 
                 structures_per_row: int = 3,
                 rows_per_page: int = 5,
                 figure_size: Tuple[float, float] = (15, 20),
                 dpi: int = 150,
                 atom_radii: float = 0.7,
                 rotation: str = '45x,-30y,10z',
                 show_unit_cell: int = 1,
                 show_legend: bool = True):
        """
        Initialize the MOF structure visualizer.
        
        Args:
            structures_per_row: Number of structures per row (default: 3)
            rows_per_page: Number of rows per page (default: 5)
            figure_size: Figure size in inches (width, height)
            dpi: Resolution for the output images
            atom_radii: Scaling factor for atom radii
            rotation: Rotation angles for 3D visualization
            show_unit_cell: Whether to show unit cell boundaries (0 or 1)
            show_legend: Whether to show atom type legend (default: True)
        """
        self.structures_per_row = structures_per_row
        self.rows_per_page = rows_per_page
        self.structures_per_page = structures_per_row * rows_per_page
        self.figure_size = figure_size
        self.dpi = dpi
        self.atom_radii = atom_radii
        self.rotation = rotation
        self.show_unit_cell = show_unit_cell
        self.show_legend = show_legend
        
    def get_cif_files(self, cif_directory: Path) -> List[Path]:
        """
        Get all CIF files from the specified directory.
        
        Args:
            cif_directory: Path to directory containing CIF files
            
        Returns:
            List of CIF file paths sorted alphabetically
        """
        if not cif_directory.exists():
            raise FileNotFoundError(f"CIF directory not found: {cif_directory}")
            
        cif_files = list(cif_directory.glob("*.cif"))
        if not cif_files:
            raise ValueError(f"No CIF files found in directory: {cif_directory}")
            
        return sorted(cif_files)
    
    def load_structure(self, cif_path: Path) -> Optional[object]:
        """
        Load MOF structure from CIF file using ASE.
        
        Args:
            cif_path: Path to CIF file
            
        Returns:
            ASE Atoms object or None if loading fails
        """
        try:
            atoms = read(str(cif_path))
            return atoms
        except Exception as e:
            print(f"Warning: Failed to load {cif_path.name}: {e}")
            return None
    
    def plot_structure(self, atoms: object, ax: plt.Axes, title: str) -> None:
        """
        Plot a single MOF structure on the given axes with atom type legend.
        
        Args:
            atoms: ASE Atoms object
            ax: Matplotlib axes object
            title: Title for the subplot
        """
        try:
            plot_atoms(atoms,
                      ax,
                      radii=self.atom_radii,
                      rotation=self.rotation,
                      show_unit_cell=self.show_unit_cell)
            
            ax.set_title(title, fontsize=12, pad=10)
            ax.set_axis_off()
            
            # Add atom type legend if enabled
            if self.show_legend:
                self.add_atom_legend(atoms, ax)
            
        except Exception as e:
            print(f"Warning: Failed to plot structure {title}: {e}")
            ax.text(0.5, 0.5, f"Error loading\n{title}", 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=8, color='red')
            ax.set_axis_off()
    
    def add_atom_legend(self, atoms: object, ax: plt.Axes) -> None:
        """
        Add a legend showing atom types and their colors.
        
        Args:
            atoms: ASE Atoms object
            ax: Matplotlib axes object
        """
        try:
            # Get unique atomic numbers and their symbols
            unique_numbers = sorted(set(atoms.get_atomic_numbers()))
            
            if len(unique_numbers) == 0:
                return
            
            # Create legend elements
            legend_elements = []
            chemical_symbols = atoms.get_chemical_symbols()
            atomic_numbers = atoms.get_atomic_numbers()
            
            for atomic_number in unique_numbers:
                # Find first occurrence of this atomic number
                index = np.where(atomic_numbers == atomic_number)[0][0]
                symbol = chemical_symbols[index]
                
                # Get color from jmol colors (same as ASE uses)
                color = jmol_colors[atomic_number]
                
                # Create circle patch for legend
                circle = Circle((0, 0), 1, facecolor=color, edgecolor='black', linewidth=0.5)
                legend_elements.append((circle, symbol))
            
            # Add legend to the plot
            if legend_elements:
                patches = [element[0] for element in legend_elements]
                labels = [element[1] for element in legend_elements]
                
                legend = ax.legend(patches, labels, 
                                 loc='upper left', 
                                 bbox_to_anchor=(1.005, 1.0),
                                 fontsize=8,
                                 frameon=True,
                                 fancybox=True,
                                 shadow=True,
                                 framealpha=0.9,
                                 ncol=1 if len(labels) <= 10 else 2)
                
                # Style the legend
                legend.get_frame().set_facecolor('white')
                legend.get_frame().set_edgecolor('gray')
                legend.get_frame().set_linewidth(0.5)
                
        except Exception as e:
            print(f"Warning: Failed to add atom legend: {e}")
    
    def create_empty_subplot(self, ax: plt.Axes) -> None:
        """
        Create an empty subplot to maintain grid layout.
        
        Args:
            ax: Matplotlib axes object
        """
        ax.set_axis_off()
        ax.text(0.5, 0.5, "", ha='center', va='center', 
               transform=ax.transAxes, fontsize=8, color='gray')
    
    def create_page(self, structures: List[Tuple[object, str]]) -> plt.Figure:
        """
        Create a single page with structures arranged in grid layout.
        
        Args:
            structures: List of (atoms, title) tuples for the page
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(self.rows_per_page, self.structures_per_row,
                                figsize=self.figure_size, dpi=self.dpi)
        
        # Ensure axes is always 2D array for consistent indexing
        if self.rows_per_page == 1 and self.structures_per_row == 1:
            # Single subplot case
            axes = np.array([[axes]])
        elif self.rows_per_page == 1:
            # Single row case
            if not isinstance(axes, np.ndarray):
                axes = np.array(axes)
            axes = axes.reshape(1, -1)
        elif self.structures_per_row == 1:
            # Single column case
            if not isinstance(axes, np.ndarray):
                axes = np.array(axes)
            axes = axes.reshape(-1, 1)
        else:
            # Multi-row, multi-column case
            if not isinstance(axes, np.ndarray):
                axes = np.array(axes)
        
        # Plot structures and fill empty spaces
        for i in range(self.rows_per_page):
            for j in range(self.structures_per_row):
                ax = axes[i, j]
                structure_index = i * self.structures_per_row + j
                
                if structure_index < len(structures):
                    atoms, title = structures[structure_index]
                    if atoms is not None:
                        self.plot_structure(atoms, ax, title)
                    else:
                        ax.text(0.5, 0.5, f"Failed to load\n{title}", 
                               ha='center', va='center', transform=ax.transAxes,
                               fontsize=8, color='red')
                        ax.set_axis_off()
                else:
                    # Fill empty subplot
                    self.create_empty_subplot(ax)
        
        # Adjust layout to accommodate external legends
        plt.tight_layout(pad=2.0, rect=[0, 0, 0.85, 1])
        return fig
    
    def visualize_structures(self, 
                           cif_directory: Path, 
                           output_pdf: Path,
                           progress_callback: Optional[callable] = None) -> None:
        """
        Generate multi-page PDF visualization of all MOF structures.
        
        Args:
            cif_directory: Path to directory containing CIF files
            output_pdf: Path for output PDF file
            progress_callback: Optional callback function for progress updates
        """
        # Get all CIF files
        cif_files = self.get_cif_files(cif_directory)
        total_files = len(cif_files)
        total_pages = math.ceil(total_files / self.structures_per_page)
        
        print(f"Found {total_files} CIF files")
        print(f"Generating {total_pages} pages with {self.structures_per_row}x{self.rows_per_page} layout")
        
        # Create output directory if it doesn't exist
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        
        with pdf_backend.PdfPages(str(output_pdf)) as pdf:
            for page_num in range(total_pages):
                start_idx = page_num * self.structures_per_page
                end_idx = min(start_idx + self.structures_per_page, total_files)
                
                print(f"Processing page {page_num + 1}/{total_pages} "
                      f"(structures {start_idx + 1}-{end_idx})")
                
                # Load structures for this page
                page_structures = []
                for i in range(start_idx, end_idx):
                    cif_file = cif_files[i]
                    atoms = self.load_structure(cif_file)
                    title = cif_file.stem  # Remove .cif extension
                    page_structures.append((atoms, title))
                    
                    if progress_callback:
                        progress_callback(i + 1, total_files)
                
                # Create and save page
                fig = self.create_page(page_structures)
                pdf.savefig(fig, bbox_inches='tight', dpi=self.dpi)
                plt.close(fig)  # Free memory
        
        print(f"Successfully generated PDF: {output_pdf}")
        print(f"Total structures processed: {total_files}")


def progress_printer(current: int, total: int) -> None:
    """Simple progress callback function."""
    if current % 10 == 0 or current == total:
        percentage = (current / total) * 100
        print(f"  Progress: {current}/{total} ({percentage:.1f}%)")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Generate multi-page PDF visualization of MOF structures from CIF files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python mof_structure_visualizer.py /path/to/cif/files output.pdf
  
  # Custom layout (4 columns, 6 rows per page)
  python mof_structure_visualizer.py /path/to/cif/files output.pdf --layout 4 6
  
  # High resolution output
  python mof_structure_visualizer.py /path/to/cif/files output.pdf --dpi 300
        """
    )
    
    parser.add_argument("cif_directory", 
                       type=Path,
                       help="Directory containing CIF files")
    
    parser.add_argument("output_pdf",
                       type=Path, 
                       help="Output PDF file path")
    
    parser.add_argument("--layout", 
                       nargs=2, 
                       type=int, 
                       default=[3, 5],
                       metavar=("COLS", "ROWS"),
                       help="Grid layout: columns and rows per page (default: 3 5)")
    
    parser.add_argument("--figure-size",
                       nargs=2,
                       type=float,
                       default=[15, 20],
                       metavar=("WIDTH", "HEIGHT"),
                       help="Figure size in inches (default: 15 20)")
    
    parser.add_argument("--dpi",
                       type=int,
                       default=150,
                       help="Output resolution in DPI (default: 150)")
    
    parser.add_argument("--atom-radii",
                       type=float,
                       default=0.7,
                       help="Atom radii scaling factor (default: 0.7)")
    
    parser.add_argument("--rotation",
                       type=str,
                       default="-5x,-5y,0z",
                       help="3D rotation angles (default: '-5x,-5y,0z')")
    
    parser.add_argument("--show-unit-cell",
                       type=int,
                       choices=[0, 1],
                       default=1,
                       help="Show unit cell boundaries: 0=no, 1=yes (default: 1)")
    
    parser.add_argument("--show-legend",
                       type=int,
                       choices=[0, 1],
                       default=1,
                       help="Show atom type legend: 0=no, 1=yes (default: 1)")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not args.cif_directory.exists():
        print(f"Error: CIF directory does not exist: {args.cif_directory}")
        sys.exit(1)
    
    if not args.cif_directory.is_dir():
        print(f"Error: CIF path is not a directory: {args.cif_directory}")
        sys.exit(1)
    
    # Initialize visualizer
    visualizer = MOFStructureVisualizer(
        structures_per_row=args.layout[0],
        rows_per_page=args.layout[1],
        figure_size=tuple(args.figure_size),
        dpi=args.dpi,
        atom_radii=args.atom_radii,
        rotation=args.rotation,
        show_unit_cell=args.show_unit_cell,
        show_legend=bool(args.show_legend)
    )
    
    try:
        # Generate visualization
        visualizer.visualize_structures(
            cif_directory=args.cif_directory,
            output_pdf=args.output_pdf,
            progress_callback=progress_printer
        )
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
