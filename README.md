# Universal 1D Heat Transfer Solver

Python tool that solves the 1D Heat Equation using the numerical Finite Difference Method (FDM). It calculates the continuous distribution of temperature across an object's thickness over time and outputs when thermal equilibrium is fully achieved.

---

## Project Description

### Overview
This project bridges thermodynamic engineering mathematics with real-world, everyday use cases. It can determine the time required to heat up:
*   **Engineering Context:** composite polymer specimens (`ePAHT-CF`) inside tensile machine furnaces for mechanical material tests.
*   **Culinary Context:** A carbon steel baking pizza stone preheating inside a home oven, or a thick beef steak heating through its thickness.

### Features
*   **Built-in Materials:** Presets for engineering polymer, carbon steel, and meat.
*   **Custom Inputs:** Enter custom physical properties (density, specific heat capacity, conductivity).
*   **Asymmetric Boundary Conditions:** Independently configure left and right environment temperatures (e.g., simulating hot pan contact on one side and cooler ambient air on the other).
*   **Animation Toggle:** Run calculations with high-fidelity live dashboard animations or switch to a background high-speed processing mode that shows only the final engineered state.
*   **Safe Folder Automation:** Automatically handles local environment paths and cleans up old image logs upon initialization.

### Prerequisites & Installation
Ensure you have Python installed alongside the required numerical libraries:
```bash
pip install numpy matplotlib