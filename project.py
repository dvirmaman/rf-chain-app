# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:31:35 2025

@author: talma
"""
import math
import pandas as pd
from itertools import permutations
import matplotlib.pyplot as plt
from itertools import accumulate
import matplotlib.patches as patches
import streamlit as st
import numpy as np




defaults = {
    "Gain": 0,
    "NF": 0,
    "IP3": float('inf'),
    "P1dB": float('inf'),
    "Z_in": 50,
    "Z_out": 50,
    "IL": 0
}

def read_parameters_from_excel(file_path):
    params_df = pd.read_excel(file_path, sheet_name="Parameters", index_col=0)
    params = params_df["Value"].to_dict()
    return params

def draw_chain(components):
    num_components = len(components)
    fig, ax = plt.subplots(figsize=(num_components * 2, 3))
    
    for i, comp in enumerate(components):
        rect = patches.Rectangle((i * 2, 1), 1.5, 1, edgecolor='black', facecolor='lightgray')
        ax.add_patch(rect)
        ax.text(i * 2 + 0.75, 1.5, comp['name'], ha='center', va='center', fontsize=12)
    
    for i in range(num_components - 1):
        ax.arrow(i * 2 + 1.5, 1.5, 0.5, 0, head_width=0.3, head_length=0.3, fc='black', ec='black')
    
    ax.set_xlim(-1, num_components * 2)
    ax.set_ylim(0, 3)
    ax.set_xticks([])
    ax.set_yticks([])
    st.pyplot(fig)




def plot_chain_metrics(chain,minimum_signal,ktb):
    """
    Plot SNR, NF, and P1dB for the components in the selected chain.
    """
    ""
    snr=[]
    total_nf = 0
    total_gain = 0
    nf_values=[]
    for i, component in enumerate(chain):
        
        gain_with_il = component['Gain'] - component.get('IL', 0)

        if i == 0:
            total_nf = component['NF']
        else:
            total_nf = 10 * math.log10(10**(total_nf / 10) + (10**(component['NF'] / 10) - 1) / (10**(total_gain / 10)))
        #print(f"total_nf: {total_nf}")

        nf_values+=[total_nf]
        if i > 0:
            prev_component = chain[i - 1]
            mismatch_loss = calculate_mismatch_loss(prev_component['Z_out'], component['Z_in'])
            total_gain -= mismatch_loss  # Subtract mismatch loss
            total_nf += mismatch_loss  # Add mismatch loss to the total NF

        total_gain += gain_with_il
        
    #print(chain)
    
    
    names = [comp['name'] for comp in chain]
    # SNR=total_gain+input_power-(10 * math.log10(ktb)+30+total_nf)
    #snr_values = list(accumulate([comp['Gain'] - comp['NF'] for comp in chain])) # Example calculation for SNR
    gain =  list(accumulate([comp['Gain'] for comp in chain]))
    p1db_values=[i + minimum_signal for i in gain] 
    nf=[i +10 * math.log10(ktb)+30 for i in nf_values] 
    SNR= [val1 - val2 for val1, val2 in zip(p1db_values, nf)]

    #print(snr_values)
    
    x = range(len(chain))
    plt.figure(figsize=(10, 6))
    plt.plot(x, SNR, label='SNR (dB)', marker='o')
    plt.plot(x, nf_values, label='NF (dB)', marker='s')
    plt.plot(x, p1db_values, label='P1dB (dBm)', marker='^')
    plt.plot(x, gain, label='gain (dBm)', marker='+')

# Add the values as text next to each point on the graph
    for i, txt in enumerate(SNR):
        plt.text(x[i], SNR[i], f"{txt:.2f}", ha='center', fontsize=10, verticalalignment='bottom')
    
    for i, txt in enumerate(nf_values):
        plt.text(x[i], nf_values[i], f"{txt:.2f}", ha='center', fontsize=10, verticalalignment='bottom')
    
    for i, txt in enumerate(p1db_values):
        plt.text(x[i], p1db_values[i], f"{txt:.2f}", ha='center', fontsize=10, verticalalignment='bottom')
    
    for i, txt in enumerate(gain):
        plt.text(x[i], gain[i], f"{txt:.2f}", ha='center', fontsize=10, verticalalignment='bottom')


    plt.xticks(x, names, rotation=45, ha='right')
    plt.xlabel('Components')
    plt.ylabel('Values (dB)')
    plt.title('RF Chain Metrics')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    st.pyplot(plt.gcf())

   
  
def read_components_from_excel(file_path):
    """
    Function to read components data from an Excel file.
    """
    #data = pd.read_excel(file_path)
    data=pd.read_excel(file_path, sheet_name="components", index_col=0)
    #print(data.columns)

    components = []
    for _, row in data.iterrows():
        component = {
            "name": row["name"] if not pd.isna(row["name"]) else "Unnamed Component",
            "Gain": float(row["Gain"]) if not pd.isna(row["Gain"]) else defaults["Gain"],
            "NF": float(row["NF"]) if not pd.isna(row["NF"]) else defaults["NF"],
            "IP3": float(row["IP3"]) if not pd.isna(row["IP3"]) else defaults["IP3"],
            "P1dB": float(row["P1dB"]) if not pd.isna(row["P1dB"]) else defaults["P1dB"],
            "Z_in": float(row["Z_in"]) if not pd.isna(row["Z_in"]) else defaults["Z_in"],
            "Z_out": float(row["Z_out"]) if not pd.isna(row["Z_out"]) else defaults["Z_out"],
            "IL": float(row["IL"]) if not pd.isna(row["IL"]) else defaults["IL"]
        }
        components.append(component)
    return components

def calculate_mismatch_loss(Z_source, Z_load):
    """
    Calculate mismatch loss (in dB) between source impedance and load impedance.
    """
    gamma = (Z_load - Z_source) / (Z_load + Z_source)  # Reflection coefficient
    mismatch_loss_db = -10 * math.log10(1 - abs(gamma)**2)  # Mismatch loss in dB
    return mismatch_loss_db

def calculate_total_nf_and_snr(chain, input_power, input_p1db,ktb):
    """
    Calculate total NF, output SNR, and check P1dB compliance for a given chain.
    """
    total_ip3=0
    total_nf = 0
    total_gain = 0
    current_power = input_power  # Start with the input power
    output_snr = 0  # Start with the input SNR
    gain_product=1

    for i, component in enumerate(chain):
        
        gain_with_il = component['Gain'] - component.get('IL', 0)

        if i == 0:
            total_nf = component['NF']
            total_ip3 = 1 / (10**(component['IP3'] / 10))
        else:
            total_nf = 10 * math.log10(10**(total_nf / 10) + (10**(component['NF'] / 10) - 1) / (10**(total_gain / 10)))
            gain_product *= 10**(total_gain / 10)
            total_ip3 += gain_product / (10**(component['IP3'] / 10)) 
            #print(f"total_nf: {total_nf}")
      
        if i > 0:
            prev_component = chain[i - 1]
            mismatch_loss = calculate_mismatch_loss(prev_component['Z_out'], component['Z_in'])
            total_gain -= mismatch_loss  # Subtract mismatch loss
            total_nf += mismatch_loss  # Add mismatch loss to the total NF

        total_gain += gain_with_il
        input_p1db += gain_with_il
        
        #output_snr = (total_gain* input_power)/(ktb*total_nf)

        if input_p1db > component['P1dB']:
            return total_nf, output_snr,total_ip3, total_gain, current_power,input_p1db, False,component['name']           # Invalid chain
   
    total_ip3 = 10 * math.log10(1 / total_ip3)
    SNR=total_gain+input_power-(10 * math.log10(ktb)+30+total_nf)
    print(f"total_ip3: {total_ip3}")
    return total_nf, SNR, total_ip3, total_gain, current_power,input_p1db, True,None

def optimize_rf_chain(components, input_power, input_p1db,ktb):
    """
    Optimize the order of RF components for the best output SNR, considering P1dB, IL, impedance matching, and IP3.
    """
    best_chain = None
    best_snr = float('inf')
    best_ip3 = float('-inf')
    all_chains = []
    for perm in permutations(components):
        total_nf, output_snr,total_ip3, total_gain, total_power, output_p1db,valid,saturation = calculate_total_nf_and_snr(perm, input_power, input_p1db,ktb)

        chain_info = {
            "chain": perm,
            "total_nf": total_nf,
            "output_snr": output_snr,
            "total_ip3" : total_ip3,
            "total_gain": total_gain,
            "output_p1db":output_p1db,
            "valid": valid,
            "saturation":saturation
            
        }
        all_chains.append(chain_info)

        if valid and ((best_snr) > (total_nf)):
            best_snr = total_nf
            best_chain = perm
            #print(f"\nbest snr: {best_snr})")

    if best_chain:
            
            st.success(f"✅ Best SNR: {output_snr:.2f} dB")
            st.success(f"✅ Total Ip3: {total_ip3:.2f} dBm")
            st.success(f"✅ Total Gain: {total_gain:.2f} dB")
            st.success(f"✅ Output power: {input_p1db:.2f} dB")
            st.success(f"✅ Total NF: {total_nf:.2f} dB")

            st.write("###  optimal chain 📊:")
            for comp in best_chain:
                st.write(f"- **{comp['name']}** | Gain: {comp['Gain']} dB | NF: {comp['NF']} dB | P1dB: {comp['P1dB']} dBm | IP3 dB: {comp['P1dB']} dBm")
            

    return best_chain

def max_input_power(chain):
    
    #print(chain)
    p_out_max = float('inf')  # נתחיל מערך גבוה מאוד
    
    for component in reversed(chain):  # מתחילים מהסוף להתחלה
         if 'P1dB' in component and component['P1dB'] is not None:
            p_in_max = component['P1dB']  # חישוב עוצמת הכניסה המרבית לרכיב
            p_out_max = min(p_out_max, p_in_max)  # לוקחים את המינימום כדי שלא תיווצר רוויה
            #print(p_out_max)
         if 'Gain' in component and component['Gain'] is not None:
             p_out_max -=component['Gain']
    print(f"max input power: {p_out_max}")
 
    #print(p_out_max)

def min_input_power(chain):

    min_output_power = min(comp["P1dB"] for comp in chain if "P1dB" in comp)  # מוצא את הרכיב עם P1dB הנמוך ביותר
    total_gain = sum(comp["Gain"] for comp in chain)  # מחבר את הרווח הכולל של השרשרת
    
    min_input = min_output_power - total_gain  # מחזיר את עוצמת הכניסה המינימלית הדרושה
    
    print(min_input)


# Read components from the Excel file
#file_path = 'SNR.xlsx'  # Path to your Excel file
#rf_chain = read_components_from_excel(file_path)
#params = read_parameters_from_excel(file_path)


st.title("RF Chain Calculator")
st.sidebar.header(" add component ➕")
if 'components' not in st.session_state:
    st.session_state.components = []
# Input fields for a new component
name = st.sidebar.text_input("name component", value="Component")
gain = st.sidebar.number_input("Gain (dB)", value=0.0)
nf = st.sidebar.number_input("Noise Figure (NF) (dB)", value=0.0)
ip3 = st.sidebar.number_input("IP3 (dBm)", value=0.0)
p1db = st.sidebar.number_input("P1dB (dBm)", value=0.0)
z_in = st.sidebar.number_input("Z_in (Ω)", value=50.0)
z_out = st.sidebar.number_input("Z_out (Ω)", value=50.0)
il = st.sidebar.number_input("Insertion Loss (dB)", value=0.0)

if st.sidebar.button("add component📥"):
    st.session_state.components.append({
        "name": name,
        "Gain": gain,
        "NF": nf,
        "IP3": ip3,
        "P1dB": p1db,
        "Z_in": z_in,
        "Z_out": z_out,
        "IL": il
    })
    st.success(f"component '{name}' add!")

rf_chain=st.session_state.components
# Button to delete all components
if st.sidebar.button("clean all components 🗑️"):
    st.session_state.components = []
    st.warning("the components deleted.")

# ✅ Displaying the entered components
if st.session_state.components:
        st.write("### 🔗 the components in the chain:")
        for i, comp in enumerate(st.session_state.components):
            st.write(
                f"**{i+1}. {comp['name']}** | Gain: {comp['Gain']} dB | NF: {comp['NF']} dB"
                f" | P1dB: {comp['P1dB']} dBm")
    # ✅ Input parameters
        st.write("## system parameters⚙️")
        input_power = st.number_input("Input Power (dBm)", value=-13.6)
        bandwidth = st.number_input("Bandwidth (Hz)", value=10e6)
        
    #input_power = params.get("input power", 0)
        minimum_signal = input_power #params.get("minimum signal", -13.6)
        B_T = bandwidth #params.get("band width", 10e6)
        k = 1.38e-23  # Boltzmann constant (J/K)
        T = 290  # Temperature in Kelvin (K)
        ktb = k*T*B_T  #ktb=k*T*B_T
        input_power=minimum_signal
 
        if st.button("optimize🚀"):
            # Optimize the RF chain
            optimized_chain = optimize_rf_chain(rf_chain, input_power, minimum_signal,ktb)
            if optimized_chain is not None:
                    plot_chain_metrics(optimized_chain,minimum_signal,ktb)
                    draw_chain(optimized_chain)
                    max_input_power(optimized_chain)
                    #print(ktb)
            else:
                    print("\nNo valid chain found due to P1dB or impedance mismatch constraints.")
                    st.error("⚠️ לא נמצאה שרשרת תקינה עקב מגבלות P1dB או חוסר התאמה בעכבות.")
else:
    st.info("⚡ add components to restatrt the buliding.")

