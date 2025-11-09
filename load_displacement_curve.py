"""
Load-Displacement Curve Analysis
Student: Aksel Pulas
Date: 2025-10-23

This script generates a professional load-displacement curve.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# TASK: Create a professional load-displacement curve
# Requirements:
# 1. Generate displacement data from 0 to 10mm
# 2. Calculate load using: load = 50 * displacement - 2 * displacement^2
# 3. Add proper labels, title, and grid
# 4. Use matplotlib with seaborn styling
# 5. Mark the yield point at displacement = 5mm

# 1. Generate displacement data from 0 to 10mm
displacement = np.linspace(0, 10, 100)  # 100 points from 0 to 10mm

# 2. Calculate load using the formula
load = 50 * displacement - 2 * displacement**2

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the load-displacement curve
plt.plot(displacement, load, linewidth=2.5, color='#2E86AB', label='Load-Displacement Curve')

# 5. Mark the yield point at displacement = 5mm
yield_displacement = 5
yield_load = 50 * yield_displacement - 2 * yield_displacement**2

plt.plot(yield_displacement, yield_load, 'ro', markersize=12, 
         label=f'Yield Point ({yield_displacement} mm, {yield_load:.1f} N)', zorder=5)

# Add annotation for yield point
plt.annotate(f'Yield Point\n({yield_displacement} mm, {yield_load:.1f} N)',
             xy=(yield_displacement, yield_load),
             xytext=(yield_displacement + 1.5, yield_load - 20),
             fontsize=11,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', 
                           color='red', lw=2))

# 3. Add proper labels, title, and grid
plt.xlabel('Displacement (mm)', fontsize=13, fontweight='bold')
plt.ylabel('Load (N)', fontsize=13, fontweight='bold')
plt.title('Professional Load-Displacement Curve', fontsize=15, fontweight='bold', pad=20)

# Add grid
plt.grid(True, alpha=0.4, linestyle='--')

# Add legend
plt.legend(loc='upper right', fontsize=11, framealpha=0.9)

# Set axis limits for better visualization
plt.xlim(-0.5, 10.5)
plt.ylim(0, max(load) * 1.1)

# Add minor grid
plt.minorticks_on()
plt.grid(which='minor', alpha=0.2, linestyle=':')

# Tight layout for better spacing
plt.tight_layout()

# Save the figure
plt.savefig('load_displacement_curve.png', dpi=300, bbox_inches='tight')
print("\n[Saved] load_displacement_curve.png")

# Show the plot
plt.show()

# Print key values
print("\n" + "="*50)
print("LOAD-DISPLACEMENT ANALYSIS RESULTS")
print("="*50)
print(f"Maximum Load: {max(load):.2f} N at {displacement[np.argmax(load)]:.2f} mm")
print(f"Yield Point: {yield_load:.2f} N at {yield_displacement} mm")
print(f"Load at 10mm: {load[-1]:.2f} N")
print("="*50)

