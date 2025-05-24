import pandas as pd
import numpy as np

def load_data_from_excel(excel_path="modul-5/modul5.xlsx"):
    """
    Loads material testing data from the specified Excel file.
    Mirrors the data loading structure from the modul5.ipynb notebook.
    """
    df_dict = {}
    sheet_names = ["kertas", "Mika"] # Case sensitive as in notebook
    
    column_map_horizontal = {
        "_horizontal_1": "A:B",
        "_horizontal_2": "E:F",
        "_horizontal_3": "I:J",
    }
    column_map_vertikal = {
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
                    # Rename columns for consistency if they are not named automatically
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
        print(f"Error: Excel file '{excel_path}' not found. Please ensure it's in the same directory.")
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

    force_raw = pd.to_numeric(dataframe.iloc[:, 0], errors='coerce')
    elongation_raw_mm = pd.to_numeric(dataframe.iloc[:, 1], errors='coerce')

    valid_indices = force_raw.notna() & elongation_raw_mm.notna()
    if not np.any(valid_indices):
        return {'E_Pa': np.nan, 'sigma_y_Pa': np.nan, 'epsilon_y': np.nan, 'Ur_J_m3': np.nan}

    force_N = force_raw[valid_indices].values
    elongation_mm = elongation_raw_mm[valid_indices].values

    if len(force_N) < 2:
        return {'E_Pa': np.nan, 'sigma_y_Pa': np.nan, 'epsilon_y': np.nan, 'Ur_J_m3': np.nan}

    # Sort by elongation
    sort_indices = np.argsort(elongation_mm)
    elongation_mm = elongation_mm[sort_indices]
    force_N = force_N[sort_indices]

    # Convert L0 to meters for strain calculation to be consistent with A0 in m^2
    L0_m = L0_mm / 1000.0
    elongation_m = elongation_mm / 1000.0

    strain = elongation_m / L0_m 
    stress_Pa = force_N / A0_m2

    # --- Young's Modulus (E) ---
    E_Pa = np.nan
    # Filter for positive strain and stress for modulus calculation, skipping first few points
    valid_modulus_points = (strain > 1e-9) & (stress_Pa > 1e-9)
    
    if np.sum(valid_modulus_points) >= 10: # Need enough points for a reliable fit
        strain_mod = strain[valid_modulus_points]
        stress_mod = stress_Pa[valid_modulus_points]
        
        # Use first few points (min 20, max 50 as in notebook, adjusted for available points)
        # Skipping the very first 5 points as in the notebook
        start_index_fit = 5
        num_total_valid_points = len(strain_mod)

        if num_total_valid_points > start_index_fit : # Ensure there are points beyond the 5th
            num_points_for_fit = min(max(15, int(num_total_valid_points * 0.1)), 45) # Use 10% of points, min 15, max 45 after skipping 5
            end_index_fit = min(start_index_fit + num_points_for_fit, num_total_valid_points)
            
            if end_index_fit - start_index_fit >= 2: # Need at least 2 points for polyfit
                elastic_strain_fit = strain_mod[start_index_fit:end_index_fit]
                elastic_stress_fit = stress_mod[start_index_fit:end_index_fit]
                
                try:
                    coeffs = np.polyfit(elastic_strain_fit, elastic_stress_fit, 1)
                    if coeffs[0] > 1e-9: # Slope must be positive
                        E_Pa = coeffs[0]
                except (np.linalg.LinAlgError, ValueError):
                    print(f"Warning: Polyfit failed for Young's Modulus calculation.")
                    E_Pa = np.nan
            else:
                 print(f"Warning: Not enough points for elastic slope after skipping initial points.")
                 E_Pa = np.nan
        else:
            print(f"Warning: Not enough valid_modulus_points beyond the initial 5 points to fit.")
            E_Pa = np.nan
    else:
        print(f"Warning: Not enough valid positive strain/stress points for modulus calculation.")

    # --- Yield Strength (sigma_y) and Yield Strain (epsilon_y) using 0.2% offset ---
    sigma_y_Pa = np.nan
    epsilon_y = np.nan
    Ur_J_m3 = np.nan
    strain_offset_val = 0.002

    if not np.isnan(E_Pa) and E_Pa > 1e-9:
        # Relevant experimental data for intersection search (strain >= 0.002)
        relevant_strain_mask = strain >= strain_offset_val
        
        if np.any(relevant_strain_mask):
            strain_relevant = strain[relevant_strain_mask]
            stress_relevant = stress_Pa[relevant_strain_mask]

            if len(strain_relevant) > 1: # Need at least two points to form a segment
                # Stress on the offset line at the experimental strain points
                stress_on_offset_line_at_exp_strains = E_Pa * (strain_relevant - strain_offset_val)
                stress_diff = stress_relevant - stress_on_offset_line_at_exp_strains
                
                # Find where the experimental stress curve crosses the offset line
                sign_changes_indices = np.where(np.diff(np.sign(stress_diff)))[0]

                if len(sign_changes_indices) > 0:
                    idx_before_crossing = sign_changes_indices[0]
                    
                    if idx_before_crossing + 1 < len(strain_relevant):
                        s1, sig1_exp = strain_relevant[idx_before_crossing], stress_relevant[idx_before_crossing]
                        s2, sig2_exp = strain_relevant[idx_before_crossing + 1], stress_relevant[idx_before_crossing + 1]
                        
                        # Linear interpolation for the experimental segment
                        m_exp = (sig2_exp - sig1_exp) / (s2 - s1) if (s2 - s1) != 0 else 0
                        c_exp = sig1_exp - m_exp * s1
                        
                        # Intersection of sigma = m_exp * epsilon + c_exp  AND  sigma = E_Pa * (epsilon - 0.002)
                        if abs(m_exp - E_Pa) < 1e-9: # Avoid division by zero (parallel lines)
                            # This case should be rare if yielding occurs.
                            # Fallback: use the point closest to the offset line
                            closest_idx_in_relevant = np.argmin(np.abs(stress_diff))
                            epsilon_y = strain_relevant[closest_idx_in_relevant]
                            sigma_y_Pa = stress_relevant[closest_idx_in_relevant]
                        else:
                            epsilon_y = (c_exp + 0.002 * E_Pa) / (E_Pa - m_exp)
                            sigma_y_Pa = E_Pa * (epsilon_y - strain_offset_val)

                        # Sanity check: yield point should be within the segment and positive stress
                        if not (min(s1, s2) <= epsilon_y <= max(s1, s2)) or sigma_y_Pa < 0:
                            # Fallback if interpolation is outside segment or yields negative stress
                            closest_idx_in_relevant = np.argmin(np.abs(stress_diff))
                            epsilon_y = strain_relevant[closest_idx_in_relevant]
                            sigma_y_Pa = max(0, stress_relevant[closest_idx_in_relevant]) # Ensure non-negative
                    else: # Crossing at the very last segment
                        epsilon_y = strain_relevant[-1]
                        sigma_y_Pa = stress_relevant[-1]
                else: # No sign change / no crossing
                    # Check if curve is always above (yielding very early or not clearly defined)
                    # or always below (no yield by this method)
                    if len(stress_diff) > 0 and np.all(stress_diff > 0): # Curve always above offset
                        # Use point with minimum positive difference as an approximation
                        min_diff_idx = np.argmin(stress_diff)
                        epsilon_y = strain_relevant[min_diff_idx]
                        sigma_y_Pa = stress_relevant[min_diff_idx]
                    # else: no yield point found by this method
            else:
                print(f"Warning: Not enough data points at or beyond 0.2% strain offset.")
        else:
            print(f"Warning: No data points at or beyond 0.2% strain offset.")


    # --- Resilience Modulus (Ur) ---
    if not np.isnan(sigma_y_Pa) and not np.isnan(E_Pa) and E_Pa > 0:
        Ur_J_m3 = (sigma_y_Pa**2) / (2 * E_Pa)
    elif not np.isnan(sigma_y_Pa) and not np.isnan(epsilon_y) and epsilon_y > 0: # Alternative if E is problematic but epsilon_y found
        Ur_J_m3 = 0.5 * sigma_y_Pa * epsilon_y


    return {
        'E_Pa': E_Pa, 
        'sigma_y_Pa': sigma_y_Pa, 
        'epsilon_y': epsilon_y, 
        'Ur_J_m3': Ur_J_m3
    }

# --- Main Execution ---

# Define initial lengths (in mm)
# From modul5.ipynb: kertas: 50mm, mika: 100mm
initial_lengths_mm = {
    "kertas": 50.0,
    "Mika": 50.0  # Matching case from notebook sheet names
}
initial_widths_mm = {
    "kertas": 30.0,  # Example width for kertas
    "Mika": 30.0     # Example width for Mika
}
initial_thickness_mm = {
    "kertas": 0.09,
    "Mika": 50.0  # Matching case from notebook sheet names
}

cross_sectional_areas_m2 = {
    "kertas": initial_widths_mm["kertas"]*initial_thickness_mm["kertas"] * 1e-6, 
    "Mika": initial_widths_mm["Mika"]*initial_thickness_mm["Mika"] * 1e-6    
}
# Default if material not in the dictionary
default_A0_m2 = 1e-6 


# Load data
data_frames = load_data_from_excel()
results_summary = {}

if data_frames:
    material_variations = {
        "kertas_horizontal": ["kertas_horizontal_1", "kertas_horizontal_2", "kertas_horizontal_3"],
        "kertas_vertikal": ["kertas_vertikal_1", "kertas_vertikal_2", "kertas_vertikal_3"],
        "Mika_horizontal": ["Mika_horizontal_1", "Mika_horizontal_2", "Mika_horizontal_3"],
        "Mika_vertikal": ["Mika_vertikal_1", "Mika_vertikal_2", "Mika_vertikal_3"],
    }

    for var_name, sample_keys in material_variations.items():
        material_type = "kertas" if "kertas" in var_name else "Mika"
        L0 = initial_lengths_mm.get(material_type)
        A0 = cross_sectional_areas_m2.get(material_type, default_A0_m2)

        if L0 is None:
            print(f"Warning: Initial length for material type '{material_type}' in variation '{var_name}' not found. Skipping.")
            results_summary[var_name] = {
                'yield_strength_Pa_avg': np.nan,
                'young_modulus_GPa_avg': np.nan,
                'resilience_modulus_J_m3_avg': np.nan,
                'notes': f'Missing L0 for {material_type}'
            }
            continue
        
        print(f"\nProcessing variation: {var_name} (L0={L0}mm, A0={A0}m^2)")

        e_moduli_pa = []
        sig_yields_pa = []
        eps_yields = []
        u_resiliences_j_m3 = []

        for sample_key in sample_keys:
            if sample_key in data_frames:
                df_sample = data_frames[sample_key]
                props = calculate_mechanical_properties(df_sample, L0, A0)
                
                e_moduli_pa.append(props['E_Pa'])
                sig_yields_pa.append(props['sigma_y_Pa'])
                eps_yields.append(props['epsilon_y']) # For reference, not directly averaged for resilience
                u_resiliences_j_m3.append(props['Ur_J_m3'])
                
                print(f"  Sample {sample_key}: E={props['E_Pa']:.2e} Pa, σ_y={props['sigma_y_Pa']:.2e} Pa, ε_y={props['epsilon_y']:.4f}, U_r={props['Ur_J_m3']:.2e} J/m³")
            else:
                print(f"  Warning: DataFrame for sample {sample_key} not found.")
                e_moduli_pa.append(np.nan)
                sig_yields_pa.append(np.nan)
                eps_yields.append(np.nan)
                u_resiliences_j_m3.append(np.nan)

        # Calculate averages, handling potential NaNs
        avg_E_Pa = np.nanmean(e_moduli_pa)
        avg_sigma_y_Pa = np.nanmean(sig_yields_pa)
        avg_Ur_J_m3 = np.nanmean(u_resiliences_j_m3)
        
        # Convert E to GPa for summary
        avg_E_GPa = avg_E_Pa / 1e9 if not np.isnan(avg_E_Pa) else np.nan

        results_summary[var_name] = {
            'yield_strength_Pa_avg': avg_sigma_y_Pa,
            'young_modulus_GPa_avg': avg_E_GPa,
            'resilience_modulus_J_m3_avg': avg_Ur_J_m3,
            'notes': 'A0 is an assumed placeholder value. Please update.' if A0 == default_A0_m2 or material_type not in cross_sectional_areas_m2 else ''
        }
        if A0 == cross_sectional_areas_m2.get(material_type, default_A0_m2) and material_type not in cross_sectional_areas_m2:
             results_summary[var_name]['notes'] += f' Used default A0={default_A0_m2} m^2.'
        elif A0 == cross_sectional_areas_m2.get(material_type):
             results_summary[var_name]['notes'] += f' Used A0_{material_type}={A0} m^2.'


# The results are stored in this dictionary
# You can print it or use it as needed in your Python environment.
# For example, to pretty print:
# import json
# print(json.dumps(results_summary, indent=4, default=lambda x: f"{x:.3e}" if isinstance(x, float) else x))

if __name__ == "__main__":
    print("\n\n--- Results Summary ---")
    if not data_frames:
        print("Could not load data. Summary is empty.")
    else:
        for var, props in results_summary.items():
            print(f"\nVariation: {var}")
            print(f"  Average Yield Strength (Pa): {props['yield_strength_Pa_avg']:.3e}" if not np.isnan(props['yield_strength_Pa_avg']) else "  Average Yield Strength (Pa): N/A")
            print(f"  Average Young's Modulus (GPa): {props['young_modulus_GPa_avg']:.3e}" if not np.isnan(props['young_modulus_GPa_avg']) else "  Average Young's Modulus (GPa): N/A")
            print(f"  Average Resilience Modulus (J/m^3): {props['resilience_modulus_J_m3_avg']:.3e}" if not np.isnan(props['resilience_modulus_J_m3_avg']) else "  Average Resilience Modulus (J/m^3): N/A")
            if props['notes']:
                print(f"  Notes: {props['notes']}")

    # To make the results_summary dictionary part of the file content when viewed,
    # it's defined above. If this script is run, it will print the summary.
    # If imported as a module, the results_summary dictionary will be available
    # after running the main part (e.g., by calling a main function that populates it).
    # For the purpose of "result in .py file", the dictionary is defined globally and populated.
    # A more typical way to "write result in .py file" would be to generate
    # a *new* .py file with this dictionary written as static data,
    # but this script makes it available dynamically upon execution.
