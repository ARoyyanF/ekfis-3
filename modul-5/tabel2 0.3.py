import pandas as pd
import numpy as np

def load_data_from_excel(excel_path="modul-5/modul5.xlsx"):
    """
    Loads material testing data from the specified Excel file.
    Mirrors the data loading structure from the modul5.ipynb notebook.
    """
    df_dict = {}
    sheet_names = ["kertas", "Mika", "Stik"] # Case sensitive as in notebook
    
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

def calculate_mechanical_properties(dataframe, L0_mm, A0_m2, material_type=None, initial_points_config=None):
    """
    Calculates Young's Modulus, Yield Strength, Yield Strain, and Resilience Modulus.
    L0_mm: Initial length in mm
    A0_m2: Cross-sectional area in m^2
    material_type: String indicating the type of material (e.g., "kertas", "Mika")
    initial_points_config: Dictionary with material-specific start/end indices for elastic fit.
                           Example: {"kertas": [10, 30]}
    """
    if dataframe.empty or len(dataframe.columns) < 2:
        return {'E_Pa': np.nan, 'sigma_y_Pa': np.nan, 'epsilon_y': np.nan, 'Ur_J_m3': np.nan}

    force_raw = pd.to_numeric(dataframe.iloc[:, 0], errors='coerce')
    elongation_raw_mm = pd.to_numeric(dataframe.iloc[:, 1], errors='coerce')

    valid_indices = force_raw.notna() & elongation_raw_mm.notna()
    if not np.any(valid_indices):
        print(f"Warning for material '{material_type if material_type else 'Unknown'}': No valid numeric data for force/elongation after cleaning.")
        return {'E_Pa': np.nan, 'sigma_y_Pa': np.nan, 'epsilon_y': np.nan, 'Ur_J_m3': np.nan}

    force_N = force_raw[valid_indices].values
    elongation_mm = elongation_raw_mm[valid_indices].values

    if len(force_N) < 2:
        print(f"Warning for material '{material_type if material_type else 'Unknown'}': Less than two valid data points after cleaning.")
        return {'E_Pa': np.nan, 'sigma_y_Pa': np.nan, 'epsilon_y': np.nan, 'Ur_J_m3': np.nan}

    sort_indices = np.argsort(elongation_mm)
    elongation_mm = elongation_mm[sort_indices]
    force_N = force_N[sort_indices]

    L0_m = L0_mm / 1000.0
    elongation_m = elongation_mm / 1000.0

    strain = elongation_m / L0_m if L0_m > 0 else np.full_like(elongation_m, np.nan)
    stress_Pa = force_N / A0_m2 if A0_m2 > 0 else np.full_like(force_N, np.nan)
    
    valid_calc_indices = ~np.isnan(strain) & ~np.isnan(stress_Pa)
    strain = strain[valid_calc_indices]
    stress_Pa = stress_Pa[valid_calc_indices]

    if len(strain) < 2:
        print(f"Warning for material '{material_type if material_type else 'Unknown'}': Less than two valid points after strain/stress calculation.")
        return {'E_Pa': np.nan, 'sigma_y_Pa': np.nan, 'epsilon_y': np.nan, 'Ur_J_m3': np.nan}

    # --- Young's Modulus (E) ---
    E_Pa = np.nan
    # Filter for positive strain and stress for modulus calculation
    valid_modulus_points = (strain > 1e-9) & (stress_Pa > 1e-9)

    # Ensure we have at least 2 points for polyfit
    if np.sum(valid_modulus_points) >= 2:
        # These are the full arrays of valid points for modulus calculation
        strain_mod_full = strain[valid_modulus_points]
        stress_mod_full = stress_Pa[valid_modulus_points]

        start_index_fit, end_index_fit = None, None # These are indices for strain_mod_full/stress_mod_full

        # Try to use material-specific range from initial_points_config
        if material_type and initial_points_config:
            mt_lower = material_type.lower() # Use lowercase for dictionary keys
            if mt_lower in initial_points_config:
                point_range = initial_points_config[mt_lower]
                slice_start_candidate = point_range[0]
                slice_end_candidate = point_range[1] # End index is exclusive for slicing

                # Ensure candidates are within the bounds of the actual data length
                if len(strain_mod_full) > 0:
                    start_index_fit_actual = max(0, slice_start_candidate)
                    start_index_fit_actual = min(start_index_fit_actual, len(strain_mod_full) - 1) # Max start index is len-1

                    end_index_fit_actual = max(0, slice_end_candidate) 
                    end_index_fit_actual = min(end_index_fit_actual, len(strain_mod_full)) # Max end index is len (exclusive)
                    
                    # Ensure at least 2 points in the slice
                    if (end_index_fit_actual - start_index_fit_actual) >= 2:
                        start_index_fit = start_index_fit_actual
                        end_index_fit = end_index_fit_actual
                    else:
                        print(f"Warning for material '{material_type}': Specified range [{slice_start_candidate},{slice_end_candidate}] resulted in an invalid/small slice [{start_index_fit_actual}:{end_index_fit_actual}] for data length {len(strain_mod_full)}. Attempting fallback.")
                        start_index_fit, end_index_fit = None, None # Reset to trigger fallback
                else: # strain_mod_full is empty
                    print(f"Warning for material '{material_type}': No valid positive strain/stress points to apply configured range. Attempting fallback.")
                    start_index_fit, end_index_fit = None, None # Reset to trigger fallback
            # else: material_type not in config, will use fallback below

        # Fallback logic if specific range not found, invalid, or resulted in too few points
        if not (start_index_fit is not None and end_index_fit is not None and (end_index_fit - start_index_fit) >= 2):
            if material_type and initial_points_config and material_type.lower() in initial_points_config and \
               not (start_index_fit is not None and end_index_fit is not None and (end_index_fit - start_index_fit) >= 2):
                # Warning about invalid slice from config already printed if it was attempted
                pass
            elif material_type and initial_points_config and not material_type.lower() in initial_points_config :
                 print(f"Warning for material '{material_type}': Not in initial_points_config. Using fallback slice for elastic region.")
            elif not material_type or not initial_points_config:
                 print(f"Warning: Material type or initial_points_config not provided. Using fallback slice for elastic region.")

            if len(strain_mod_full) >= 2: # Check if enough data for any fallback
                fs_fallback = 5  # fallback_start_index
                fe_fallback = 20 # fallback_end_index (exclusive)
                
                fs_fallback_actual = min(fs_fallback, len(strain_mod_full) - 2) 
                if fs_fallback_actual < 0: fs_fallback_actual = 0

                fe_fallback_actual = min(fe_fallback, len(strain_mod_full))
                if fe_fallback_actual <= fs_fallback_actual + 1 : 
                    fe_fallback_actual = fs_fallback_actual + 2 
                    fe_fallback_actual = min(fe_fallback_actual, len(strain_mod_full))

                if (fe_fallback_actual - fs_fallback_actual) >= 2:
                    start_index_fit = fs_fallback_actual
                    end_index_fit = fe_fallback_actual
                    print(f"Info for material '{material_type if material_type else 'Unknown'}': Using fallback elastic region slice [{start_index_fit}:{end_index_fit}].")
                else: 
                    start_index_fit, end_index_fit = None, None
            else: # Not enough points in strain_mod_full for any fallback
                start_index_fit, end_index_fit = None, None
            
            if not (start_index_fit is not None and end_index_fit is not None and (end_index_fit - start_index_fit) >= 2):
                 print(f"Warning for material '{material_type if material_type else 'Unknown'}': Fallback slice logic also resulted in < 2 points. Cannot fit elastic slope.")

        # Perform fit if a valid slice was determined
        if start_index_fit is not None and end_index_fit is not None and (end_index_fit - start_index_fit) >= 2:
            elastic_strain_fit = strain_mod_full[start_index_fit:end_index_fit]
            elastic_stress_fit = stress_mod_full[start_index_fit:end_index_fit]
            
            try:
                coeffs = np.polyfit(elastic_strain_fit, elastic_stress_fit, 1)
                if coeffs[0] > 1e-9: 
                    E_Pa = coeffs[0]
            except (np.linalg.LinAlgError, ValueError) as e:
                print(f"Warning: Polyfit failed for Young's Modulus ({material_type if material_type else 'Unknown'}). Error: {e}")
                E_Pa = np.nan
        else:
             print(f"Warning: Not enough points for elastic slope fit for material '{material_type if material_type else 'Unknown'}' after all checks.")
             E_Pa = np.nan
    else: # np.sum(valid_modulus_points) < 2
        print(f"Warning: Less than two valid positive strain/stress points for modulus calculation (material: {material_type if material_type else 'Unknown'}). Found {np.sum(valid_modulus_points)}.")
        E_Pa = np.nan

    # --- Yield Strength (sigma_y) and Yield Strain (epsilon_y) using 0.2% offset method ---
    sigma_y_Pa = np.nan
    epsilon_y = np.nan
    Ur_J_m3 = np.nan 
    strain_offset_val = 0.002 

    if not np.isnan(E_Pa) and E_Pa > 1e-9: 
        relevant_strain_mask = strain >= strain_offset_val
        
        if np.any(relevant_strain_mask):
            strain_relevant = strain[relevant_strain_mask]
            stress_relevant = stress_Pa[relevant_strain_mask]

            if len(strain_relevant) > 1: 
                stress_on_offset_line_at_exp_strains = E_Pa * (strain_relevant - strain_offset_val)
                stress_diff = stress_relevant - stress_on_offset_line_at_exp_strains
                
                sign_changes_indices = np.where(np.diff(np.sign(stress_diff)))[0]

                if len(sign_changes_indices) > 0:
                    idx_before_crossing = sign_changes_indices[0] 
                    
                    if idx_before_crossing + 1 < len(strain_relevant):
                        s1, sig1_exp = strain_relevant[idx_before_crossing], stress_relevant[idx_before_crossing]
                        s2, sig2_exp = strain_relevant[idx_before_crossing + 1], stress_relevant[idx_before_crossing + 1]
                        
                        m_exp = (sig2_exp - sig1_exp) / (s2 - s1) if (s2 - s1) != 0 else 0
                        c_exp = sig1_exp - m_exp * s1
                        
                        if abs(m_exp - E_Pa) < 1e-9: 
                            closest_idx_in_relevant = np.argmin(np.abs(stress_diff))
                            epsilon_y = strain_relevant[closest_idx_in_relevant]
                            sigma_y_Pa = stress_relevant[closest_idx_in_relevant]
                        else:
                            epsilon_y = (c_exp + strain_offset_val * E_Pa) / (E_Pa - m_exp)
                            sigma_y_Pa = E_Pa * (epsilon_y - strain_offset_val)

                        if not (min(s1, s2) <= epsilon_y <= max(s1, s2)) or sigma_y_Pa < 0:
                            closest_idx_in_relevant = np.argmin(np.abs(stress_diff))
                            epsilon_y = strain_relevant[closest_idx_in_relevant]
                            sigma_y_Pa = max(0, stress_relevant[closest_idx_in_relevant]) 
                    else: 
                        epsilon_y = strain_relevant[-1]
                        sigma_y_Pa = stress_relevant[-1]
                else: 
                    if len(stress_diff) > 0 and np.all(stress_diff > 0): 
                        min_diff_idx = np.argmin(stress_diff) 
                        epsilon_y = strain_relevant[min_diff_idx]
                        sigma_y_Pa = stress_relevant[min_diff_idx]
            else:
                print(f"Warning for material '{material_type if material_type else 'Unknown'}': Not enough relevant data points (strain >= {strain_offset_val}) for yield strength calculation.")
        else:
            print(f"Warning for material '{material_type if material_type else 'Unknown'}': No data points at or beyond {strain_offset_val*100}% strain offset.")

    # --- Resilience Modulus (Ur) ---
    if not np.isnan(sigma_y_Pa) and not np.isnan(E_Pa) and E_Pa > 0:
        Ur_J_m3 = (sigma_y_Pa**2) / (2 * E_Pa)
    elif not np.isnan(sigma_y_Pa) and not np.isnan(epsilon_y) and epsilon_y > 0:
        Ur_J_m3 = 0.5 * sigma_y_Pa * epsilon_y

    return {
        'E_Pa': E_Pa, 
        'sigma_y_Pa': sigma_y_Pa, 
        'epsilon_y': epsilon_y, 
        'Ur_J_m3': Ur_J_m3
    }

# --- Main Execution ---

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

# Configuration for elastic region fitting, similar to initial_points_range in modul5.ipynb
# Keys should be lowercase material names. Values are [start_index, end_index (exclusive)]
initial_points_range_dict = { 
    "kertas": [10, 30],
    "mika": [10, 60], # Note: "mika" is lowercase to match notebook's dictionary key convention
    "stik": [0, 20],
}


data_frames = load_data_from_excel() 
results_summary = {}

if data_frames:
    material_variations = {
        "kertas_horizontal": ["kertas_horizontal_1", "kertas_horizontal_2", "kertas_horizontal_3"],
        "kertas_vertikal": ["kertas_vertikal_1", "kertas_vertikal_2", "kertas_vertikal_3"],
        "Mika_horizontal": ["Mika_horizontal_1", "Mika_horizontal_2", "Mika_horizontal_3"],
        "Mika_vertikal": ["Mika_vertikal_1", "Mika_vertikal_2", "Mika_vertikal_3"],
        "Stik_horizontal": ["Stik_horizontal_1"], # Assuming only one sample for Stik based on original script
        "Stik_vertikal": ["Stik_vertikal_1"],   # Assuming only one sample for Stik
    }

    for var_name, sample_keys in material_variations.items():
        material_type = None # Will be "kertas", "Mika", or "Stik"
        
        if "kertas" in var_name.lower(): # Use lower() for robust matching
            material_type = "kertas"
        elif "mika" in var_name.lower(): # Use lower()
            material_type = "Mika" # Keep original casing for dimension dicts if needed, but pass lower to function
        elif "stik" in var_name.lower(): # Use lower()
             material_type = "Stik"
        else:
            print(f"Warning: Unknown material type in variation '{var_name}'. Skipping.")
            results_summary[var_name] = {
                'yield_strength_Pa_avg': np.nan, 'young_modulus_GPa_avg': np.nan,
                'resilience_modulus_J_m3_avg': np.nan, 'notes': f'Unknown material in {var_name}'
            }
            continue
        
        orientation = None
        if "horizontal" in var_name.lower():
            orientation = "horizontal"
            # Use material_type (which could be "Mika" or "kertas" or "Stik") for dimension dicts
            L0_mm = initial_lengths_horizontal_mm.get(material_type)
            width_mm = initial_widths_horizontal_mm.get(material_type)
            thickness_mm = initial_thickness_horizontal_mm.get(material_type)
        elif "vertikal" in var_name.lower():
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
        
        A0_m2 = (width_mm / 1000.0) * (thickness_mm / 1000.0) 

        print(f"\nProcessing variation: {var_name} (Material: {material_type}, Orientation: {orientation})")
        print(f"  L0={L0_mm}mm, Width={width_mm}mm, Thickness={thickness_mm}mm, A0={A0_m2:.3e}m^2")

        e_moduli_pa = []
        sig_yields_pa = []
        u_resiliences_j_m3 = []

        for sample_key in sample_keys:
            if sample_key in data_frames:
                df_sample = data_frames[sample_key]
                # Pass material_type and the config dictionary
                props = calculate_mechanical_properties(
                    df_sample, 
                    L0_mm, 
                    A0_m2, 
                    material_type=material_type, # Pass the determined material_type
                    initial_points_config=initial_points_range_dict # Pass the config dict
                )
                
                e_moduli_pa.append(props['E_Pa'])
                sig_yields_pa.append(props['sigma_y_Pa'])
                u_resiliences_j_m3.append(props['Ur_J_m3'])
                
                print(f"  Sample {sample_key}: E={props['E_Pa']:.2e} Pa, σ_y={props['sigma_y_Pa']:.2e} Pa, ε_y={props['epsilon_y']:.4f}, U_r={props['Ur_J_m3']:.2e} J/m³")
            else:
                print(f"  Warning: DataFrame for sample {sample_key} not found.")
                e_moduli_pa.append(np.nan)
                sig_yields_pa.append(np.nan)
                u_resiliences_j_m3.append(np.nan)

        avg_E_Pa = np.nanmean(e_moduli_pa) if len(e_moduli_pa) > 0 else np.nan
        avg_sigma_y_Pa = np.nanmean(sig_yields_pa) if len(sig_yields_pa) > 0 else np.nan
        avg_Ur_J_m3 = np.nanmean(u_resiliences_j_m3) if len(u_resiliences_j_m3) > 0 else np.nan
        
        avg_E_GPa = avg_E_Pa / 1e9 if not np.isnan(avg_E_Pa) else np.nan

        results_summary[var_name] = {
            'yield_strength_Pa_avg': avg_sigma_y_Pa,
            'young_modulus_GPa_avg': avg_E_GPa,
            'resilience_modulus_J_m3_avg': avg_Ur_J_m3,
            'notes': f"A0_{material_type}={A0_m2} m^2"
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
            ys_str = f"{props['yield_strength_Pa_avg']:.3e}" if not np.isnan(props['yield_strength_Pa_avg']) else "N/A"
            ym_str = f"{props['young_modulus_GPa_avg']:.3e}" if not np.isnan(props['young_modulus_GPa_avg']) else "N/A"
            rm_str = f"{props['resilience_modulus_J_m3_avg']:.3e}" if not np.isnan(props['resilience_modulus_J_m3_avg']) else "N/A"
            
            print(f"  Average Yield Strength (Pa): {ys_str}")
            print(f"  Average Young's Modulus (GPa): {ym_str}")
            print(f"  Average Resilience Modulus (J/m^3): {rm_str}")
            if props['notes']:
                print(f"  Notes: {props['notes']}")
