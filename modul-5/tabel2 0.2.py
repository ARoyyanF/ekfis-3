import pandas as pd
import numpy as np

def load_data_from_excel(excel_path="modul-5/modul5.xlsx"):
    """
    Loads material testing data from the specified Excel file.
    Mirrors the data loading structure from the modul5.ipynb notebook.
    """
    df_dict = {}
    # Defines which sheets to load from the Excel file.
    # Currently, only "kertas" and "Mika" are loaded.
    # To process "Stik", add "Stik" to this list and ensure a corresponding sheet exists in the Excel.
    sheet_names = ["kertas", "Mika", "Stik"] # Case sensitive as in notebook
    
    column_map_horizontal = {
        "_horizontal_1": "A:B",
        "_horizontal_2": "E:F",
        "_horizontal_3": "I:J",
    }
    column_map_vertikal = { # Using "vertikal" to match existing script conventions
        "_vertikal_1": "U:V",
        "_vertikal_2": "Y:Z",
        "_vertikal_3": "AC:AD",
    }

    try:
        xls = pd.ExcelFile(excel_path)
        for sheet_name in sheet_names:
            if sheet_name not in xls.sheet_names:
                print(f"Warning: Sheet '{sheet_name}' not found in '{excel_path}'. Skipping.")
                continue
            
            # Horizontal samples
            for suffix, cols in column_map_horizontal.items():
                key = sheet_name + suffix
                try:
                    df_dict[key] = pd.read_excel(xls, sheet_name=sheet_name, skiprows=1, usecols=cols)
                    if len(df_dict[key].columns) == 2:
                        df_dict[key].columns = ['Force', 'Elongation']
                except ValueError as e:
                    print(f"Warning: Could not load columns {cols} for {key}. Error: {e}")
                    df_dict[key] = pd.DataFrame(columns=['Force', 'Elongation'])

            # Vertikal samples
            for suffix, cols in column_map_vertikal.items():
                key = sheet_name + suffix
                try:
                    df_dict[key] = pd.read_excel(xls, sheet_name=sheet_name, skiprows=1, usecols=cols)
                    if len(df_dict[key].columns) == 2:
                         df_dict[key].columns = ['Force', 'Elongation']
                except ValueError as e:
                    print(f"Warning: Could not load columns {cols} for {key}. Error: {e}")
                    df_dict[key] = pd.DataFrame(columns=['Force', 'Elongation'])
                    
    except FileNotFoundError:
        print(f"Error: Excel file '{excel_path}' not found. Please ensure it's in the correct path.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the Excel file: {e}")
        return None
        
    return df_dict

def calculate_mechanical_properties(dataframe, L0_mm, A0_m2):
    """
    Calculates Young's Modulus, Yield Strength, Yield Strain, and Resilience Modulus.
    L0_mm: Initial length in mm
    A0_m2: Cross-sectional area in m^2
    """
    if dataframe.empty or len(dataframe.columns) < 2:
        return {'E_Pa': np.nan, 'sigma_y_Pa': np.nan, 'epsilon_y': np.nan, 'Ur_J_m3': np.nan}

    # Attempt to convert force and elongation to numeric, coercing errors to NaN
    force_raw = pd.to_numeric(dataframe.iloc[:, 0], errors='coerce')
    elongation_raw_mm = pd.to_numeric(dataframe.iloc[:, 1], errors='coerce')

    # Filter out rows where either force or elongation is NaN after conversion
    valid_indices = force_raw.notna() & elongation_raw_mm.notna()
    if not np.any(valid_indices): # Check if any valid data remains
        print("Warning: No valid numeric data for force/elongation after cleaning.")
        return {'E_Pa': np.nan, 'sigma_y_Pa': np.nan, 'epsilon_y': np.nan, 'Ur_J_m3': np.nan}

    force_N = force_raw[valid_indices].values
    elongation_mm = elongation_raw_mm[valid_indices].values

    if len(force_N) < 2: # Need at least two points for calculations
        print("Warning: Less than two valid data points after cleaning.")
        return {'E_Pa': np.nan, 'sigma_y_Pa': np.nan, 'epsilon_y': np.nan, 'Ur_J_m3': np.nan}

    # Sort data by elongation to ensure correct order for calculations
    sort_indices = np.argsort(elongation_mm)
    elongation_mm = elongation_mm[sort_indices]
    force_N = force_N[sort_indices]

    # Convert initial length (L0) and elongation to meters for consistent unit calculations
    L0_m = L0_mm / 1000.0
    elongation_m = elongation_mm / 1000.0

    # Calculate strain and stress
    # Avoid division by zero if L0_m is zero (though L0_mm should be > 0)
    strain = elongation_m / L0_m if L0_m > 0 else np.full_like(elongation_m, np.nan)
    stress_Pa = force_N / A0_m2 if A0_m2 > 0 else np.full_like(force_N, np.nan)
    
    # Remove NaNs that might have been introduced if L0_m or A0_m2 was zero/invalid
    valid_calc_indices = ~np.isnan(strain) & ~np.isnan(stress_Pa)
    strain = strain[valid_calc_indices]
    stress_Pa = stress_Pa[valid_calc_indices]

    if len(strain) < 2:
        print("Warning: Less than two valid points after strain/stress calculation.")
        return {'E_Pa': np.nan, 'sigma_y_Pa': np.nan, 'epsilon_y': np.nan, 'Ur_J_m3': np.nan}

    # --- Young's Modulus (E) ---
    E_Pa = np.nan
    # Filter for positive strain and stress for modulus calculation
    # Using a small epsilon to avoid issues with floating point comparisons to zero
    valid_modulus_points = (strain > 1e-9) & (stress_Pa > 1e-9) 
    
    if np.sum(valid_modulus_points) >= 10: # Need enough points for a reliable fit
        strain_mod = strain[valid_modulus_points]
        stress_mod = stress_Pa[valid_modulus_points]
        
        # Skip initial points and select a portion for linear fit (elastic region)
        start_index_fit = 5 
        num_total_valid_points = len(strain_mod)

        if num_total_valid_points > start_index_fit :
            # Use 10% of points, with a min of 15 and max of 45, after skipping the first 5
            num_points_for_fit = min(max(15, int(num_total_valid_points * 0.1)), 45) 
            end_index_fit = min(start_index_fit + num_points_for_fit, num_total_valid_points)
            
            if end_index_fit - start_index_fit >= 2: # Need at least 2 points for polyfit
                elastic_strain_fit = strain_mod[start_index_fit:end_index_fit]
                elastic_stress_fit = stress_mod[start_index_fit:end_index_fit]
                
                try:
                    # Linear fit (degree 1 polynomial)
                    coeffs = np.polyfit(elastic_strain_fit, elastic_stress_fit, 1)
                    if coeffs[0] > 1e-9: # Slope (Young's Modulus) must be positive
                        E_Pa = coeffs[0]
                except (np.linalg.LinAlgError, ValueError) as e:
                    print(f"Warning: Polyfit failed for Young's Modulus. Error: {e}")
                    E_Pa = np.nan
            else:
                 print(f"Warning: Not enough points for elastic slope fit after skipping initial points.")
                 E_Pa = np.nan
        else:
            print(f"Warning: Not enough valid_modulus_points beyond the initial 5 to fit E.")
            E_Pa = np.nan
    else:
        print(f"Warning: Not enough valid positive strain/stress points for modulus calculation (need >=10). Found {np.sum(valid_modulus_points)}.")

    # --- Yield Strength (sigma_y) and Yield Strain (epsilon_y) using 0.2% offset method ---
    sigma_y_Pa = np.nan
    epsilon_y = np.nan
    Ur_J_m3 = np.nan # Resilience Modulus
    strain_offset_val = 0.002 # 0.2% strain offset

    if not np.isnan(E_Pa) and E_Pa > 1e-9: # Proceed if Young's Modulus is valid
        # Consider experimental data points where strain is at or beyond the offset value
        relevant_strain_mask = strain >= strain_offset_val
        
        if np.any(relevant_strain_mask):
            strain_relevant = strain[relevant_strain_mask]
            stress_relevant = stress_Pa[relevant_strain_mask]

            if len(strain_relevant) > 1: 
                # Calculate stress values on the offset line corresponding to experimental strain points
                stress_on_offset_line_at_exp_strains = E_Pa * (strain_relevant - strain_offset_val)
                # Difference between experimental stress and offset line stress
                stress_diff = stress_relevant - stress_on_offset_line_at_exp_strains
                
                # Find indices where the sign of stress_diff changes (indicating a crossing)
                sign_changes_indices = np.where(np.diff(np.sign(stress_diff)))[0]

                if len(sign_changes_indices) > 0:
                    idx_before_crossing = sign_changes_indices[0] # First crossing
                    
                    if idx_before_crossing + 1 < len(strain_relevant):
                        # Points bracketing the intersection
                        s1, sig1_exp = strain_relevant[idx_before_crossing], stress_relevant[idx_before_crossing]
                        s2, sig2_exp = strain_relevant[idx_before_crossing + 1], stress_relevant[idx_before_crossing + 1]
                        
                        # Linear interpolation for the experimental stress-strain segment
                        m_exp = (sig2_exp - sig1_exp) / (s2 - s1) if (s2 - s1) != 0 else 0
                        c_exp = sig1_exp - m_exp * s1
                        
                        # Intersection of: sigma = m_exp * epsilon + c_exp  (experimental)
                        # AND             sigma = E_Pa * (epsilon - 0.002) (offset line)
                        if abs(m_exp - E_Pa) < 1e-9: # Avoid division by near-zero (parallel lines)
                            # Fallback: use the point closest to the offset line if lines are parallel
                            closest_idx_in_relevant = np.argmin(np.abs(stress_diff))
                            epsilon_y = strain_relevant[closest_idx_in_relevant]
                            sigma_y_Pa = stress_relevant[closest_idx_in_relevant]
                        else:
                            epsilon_y = (c_exp + strain_offset_val * E_Pa) / (E_Pa - m_exp)
                            sigma_y_Pa = E_Pa * (epsilon_y - strain_offset_val)

                        # Sanity check: yield point should be within the segment and have positive stress
                        if not (min(s1, s2) <= epsilon_y <= max(s1, s2)) or sigma_y_Pa < 0:
                            # Fallback if interpolation is outside segment or yields negative stress
                            closest_idx_in_relevant = np.argmin(np.abs(stress_diff))
                            epsilon_y = strain_relevant[closest_idx_in_relevant]
                            sigma_y_Pa = max(0, stress_relevant[closest_idx_in_relevant]) # Ensure non-negative
                    else: # Crossing detected at the very last segment
                        epsilon_y = strain_relevant[-1]
                        sigma_y_Pa = stress_relevant[-1]
                else: # No sign change / no crossing found
                    # If experimental curve is always above the offset line (early yield or no clear yield by this method)
                    if len(stress_diff) > 0 and np.all(stress_diff > 0): 
                        min_diff_idx = np.argmin(stress_diff) # Point closest to offset line
                        epsilon_y = strain_relevant[min_diff_idx]
                        sigma_y_Pa = stress_relevant[min_diff_idx]
            else:
                print(f"Warning: Not enough relevant data points (strain >= {strain_offset_val}) for yield strength calculation.")
        else:
            print(f"Warning: No data points at or beyond {strain_offset_val*100}% strain offset.")

    # --- Resilience Modulus (Ur) ---
    # Calculated if yield strength and Young's modulus are valid
    if not np.isnan(sigma_y_Pa) and not np.isnan(E_Pa) and E_Pa > 0:
        Ur_J_m3 = (sigma_y_Pa**2) / (2 * E_Pa)
    # Alternative calculation if E_Pa is problematic but yield strain was found
    elif not np.isnan(sigma_y_Pa) and not np.isnan(epsilon_y) and epsilon_y > 0:
        Ur_J_m3 = 0.5 * sigma_y_Pa * epsilon_y

    return {
        'E_Pa': E_Pa, 
        'sigma_y_Pa': sigma_y_Pa, 
        'epsilon_y': epsilon_y, 
        'Ur_J_m3': Ur_J_m3
    }

# --- Main Execution ---

# Define initial dimensions (in mm) - REPLACED WITH USER'S NEW VALUES
initial_lengths_vertical_mm = {
    "kertas": 50.0,
    "Mika": 50.0,
    "Stik": 2.2
}
initial_lengths_horizontal_mm = {
    "kertas": 50.0,
    "Mika": 50.0,
    "Stik": 1.0
}
initial_widths_horizontal_mm = {
    "kertas": 30.0,
    "Mika": 30.0,
    "Stik": 2.2
}
initial_widths_vertical_mm = {
    "kertas": 30.0,
    "Mika": 30.0,
    "Stik": 1.0
}
initial_thickness_horizontal_mm = {
    "kertas": 0.04,
    "Mika": 0.07,
    "Stik": 1.425
}
initial_thickness_vertical_mm = {
    "kertas": 0.09,
    "Mika": 0.08,
    "Stik": 1.425
}

# Load data from Excel
data_frames = load_data_from_excel() # Ensure "modul-5/modul5.xlsx" is the correct path
results_summary = {}

if data_frames:
    # Defines which sets of samples to process.
    # To process "Stik", add entries like "Stik_horizontal", "Stik_vertikal"
    # and ensure "Stik" is in sheet_names in load_data_from_excel.
    material_variations = {
        "kertas_horizontal": ["kertas_horizontal_1", "kertas_horizontal_2", "kertas_horizontal_3"],
        "kertas_vertikal": ["kertas_vertikal_1", "kertas_vertikal_2", "kertas_vertikal_3"],
        "Mika_horizontal": ["Mika_horizontal_1", "Mika_horizontal_2", "Mika_horizontal_3"],
        "Mika_vertikal": ["Mika_vertikal_1", "Mika_vertikal_2", "Mika_vertikal_3"],
        "Stik_horizontal": ["Stik_horizontal_1"],
        "Stik_vertikal": ["Stik_vertikal_1"],
    }

    for var_name, sample_keys in material_variations.items():
        material_type = None
        orientation = None

        if "kertas" in var_name:
            material_type = "kertas"
        elif "Mika" in var_name: # Ensure case matches keys in dimension dictionaries
            material_type = "Mika"
        elif "Stik" in var_name: # For future use if Stik variations are added
             material_type = "Stik"
        else:
            print(f"Warning: Unknown material type in variation '{var_name}'. Skipping.")
            results_summary[var_name] = {
                'yield_strength_Pa_avg': np.nan, 'young_modulus_GPa_avg': np.nan,
                'resilience_modulus_J_m3_avg': np.nan, 'notes': f'Unknown material in {var_name}'
            }
            continue
        
        if "horizontal" in var_name:
            orientation = "horizontal"
            L0_mm = initial_lengths_horizontal_mm.get(material_type)
            width_mm = initial_widths_horizontal_mm.get(material_type)
            thickness_mm = initial_thickness_horizontal_mm.get(material_type)
        elif "vertikal" in var_name:
            orientation = "vertikal"
            L0_mm = initial_lengths_vertical_mm.get(material_type)
            width_mm = initial_widths_vertical_mm.get(material_type)
            thickness_mm = initial_thickness_vertical_mm.get(material_type)
        else:
            print(f"Warning: Could not determine orientation for '{var_name}'. Skipping.")
            results_summary[var_name] = {
                'yield_strength_Pa_avg': np.nan, 'young_modulus_GPa_avg': np.nan,
                'resilience_modulus_J_m3_avg': np.nan, 'notes': f'Unknown orientation in {var_name}'
            }
            continue

        if L0_mm is None or width_mm is None or thickness_mm is None:
            print(f"Warning: Missing dimensions for material '{material_type}' ({orientation}) in variation '{var_name}'. Skipping.")
            results_summary[var_name] = {
                'yield_strength_Pa_avg': np.nan, 'young_modulus_GPa_avg': np.nan,
                'resilience_modulus_J_m3_avg': np.nan, 
                'notes': f'Missing L0/width/thickness for {material_type} ({orientation})'
            }
            continue
        
        # Calculate cross-sectional area (A0) in m^2
        A0_m2 = (width_mm / 1000.0) * (thickness_mm / 1000.0) # width_m * thickness_m
        # Or, A0_m2 = width_mm * thickness_mm * 1e-6 if keeping them in mm then converting area

        print(f"\nProcessing variation: {var_name} (Material: {material_type}, Orientation: {orientation})")
        print(f"  L0={L0_mm}mm, Width={width_mm}mm, Thickness={thickness_mm}mm, A0={A0_m2:.3e}m^2")

        e_moduli_pa = []
        sig_yields_pa = []
        # eps_yields = [] # Not directly averaged for resilience, can be kept for individual sample output if needed
        u_resiliences_j_m3 = []

        for sample_key in sample_keys:
            if sample_key in data_frames:
                df_sample = data_frames[sample_key]
                props = calculate_mechanical_properties(df_sample, L0_mm, A0_m2)
                
                e_moduli_pa.append(props['E_Pa'])
                sig_yields_pa.append(props['sigma_y_Pa'])
                # eps_yields.append(props['epsilon_y'])
                u_resiliences_j_m3.append(props['Ur_J_m3'])
                
                print(f"  Sample {sample_key}: E={props['E_Pa']:.2e} Pa, σ_y={props['sigma_y_Pa']:.2e} Pa, ε_y={props['epsilon_y']:.4f}, U_r={props['Ur_J_m3']:.2e} J/m³")
            else:
                print(f"  Warning: DataFrame for sample {sample_key} not found.")
                e_moduli_pa.append(np.nan)
                sig_yields_pa.append(np.nan)
                # eps_yields.append(np.nan)
                u_resiliences_j_m3.append(np.nan)

        # Calculate averages, handling potential NaNs
        avg_E_Pa = np.nanmean(e_moduli_pa) if len(e_moduli_pa) > 0 else np.nan
        avg_sigma_y_Pa = np.nanmean(sig_yields_pa) if len(sig_yields_pa) > 0 else np.nan
        avg_Ur_J_m3 = np.nanmean(u_resiliences_j_m3) if len(u_resiliences_j_m3) > 0 else np.nan
        
        # Convert E to GPa for summary
        avg_E_GPa = avg_E_Pa / 1e9 if not np.isnan(avg_E_Pa) else np.nan

        results_summary[var_name] = {
            'yield_strength_Pa_avg': avg_sigma_y_Pa,
            'young_modulus_GPa_avg': avg_E_GPa,
            'resilience_modulus_J_m3_avg': avg_Ur_J_m3,
            'notes': '' # Notes are now more specific if errors occur during dimension lookup
        }

if __name__ == "__main__":
    print("\n\n--- Results Summary ---")
    if not data_frames:
        print("Could not load data from Excel. Summary is empty.")
    elif not results_summary:
        print("Data loaded, but no variations were processed or all resulted in errors. Summary is empty.")
    else:
        for var, props in results_summary.items():
            print(f"\nVariation: {var}")
            # Check for NaN before formatting to avoid "nan" string, show "N/A" instead
            ys_str = f"{props['yield_strength_Pa_avg']:.3e}" if not np.isnan(props['yield_strength_Pa_avg']) else "N/A"
            ym_str = f"{props['young_modulus_GPa_avg']:.3e}" if not np.isnan(props['young_modulus_GPa_avg']) else "N/A"
            rm_str = f"{props['resilience_modulus_J_m3_avg']:.3e}" if not np.isnan(props['resilience_modulus_J_m3_avg']) else "N/A"
            
            print(f"  Average Yield Strength (Pa): {ys_str}")
            print(f"  Average Young's Modulus (GPa): {ym_str}")
            print(f"  Average Resilience Modulus (J/m^3): {rm_str}")
            if props['notes']: # Print notes if any were added (e.g., for missing dimensions)
                print(f"  Notes: {props['notes']}")
