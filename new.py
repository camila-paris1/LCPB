import os
import uproot
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def list_tree_branches(file_path: str):
    with uproot.open(file_path) as file:
        tree = file[file.keys()[0]]
        return tree.keys()

def load_ntuple_data(folder_name: str, chunk_name: str, branch_name: str) -> pd.DataFrame:
    file_path = os.path.join(folder_name, chunk_name)
    with uproot.open(file_path) as file:
        tree = file[file.keys()[0]]
        df = tree.arrays([branch_name], library='pd')
    return df


folder =[
    "p8_ee_tt_ecm365",
    "p8_ee_WW_ecm365",
    "p8_ee_ZZ_ecm365",
    "wzp6_ee_bbH_ecm365",
    "wzp6_ee_ccH_ecm365",
    "wzp6_ee_eeH_ecm365",
    "wzp6_ee_mumuH_ecm365",
    "wzp6_ee_SM_tt_tWbTWs_tallTheavy_ecm365",
    "wzp6_ee_SM_tt_tWsTWb_theavyTall_ecm365"
]

chunk04 = [
    "chunk0.root",
    "chunk1.root",
    "chunk2.root",
    "chunk3.root",
    "chunk4.root"
]

chunk05 = [
    "chunk0.root",
    "chunk1.root",
    "chunk2.root",
    "chunk3.root",
    "chunk4.root",
    "chunk5.root"
]

branch = [
    "n_genTops",
    "n_genWs",
    "n_genBottoms",
    "n_genMuons",
    "n_genElectrons",
    "Wp_elenu",
    "Wp_munu",
    "Wp_taunu",
    "Wm_elenu",
    "Wm_munu",
    "Wm_taunu",
    "genTop_px",
    "genTop_py",
    "genTop_pz",
    "genTop_phi",
    "genTop_eta",
    "genTop_energy",
    "genTop_mass",
    "genTop_pdg",
    "genBottom_px",
    "genBottom_py",
    "genBottom_pz",
    "genBottom_phi",
    "genBottom_eta",
    "genBottom_energy",
    "genBottom_mass",
    "genBottom_pdg",
    "genW_px",
    "genW_py",
    "genW_pz",
    "genW_phi",
    "genW_eta",
    "genW_energy",
    "genW_mass",
    "genW_charge",
    "genMuon_px",
    "genMuon_py",
    "genMuon_pz",
    "genMuon_phi",
    "genMuon_eta",
    "genMuon_energy",
    "genMuon_mass",
    "genMuon_charge",
    "genMuon_parentPDG",
    "genElectron_px",
    "genElectron_py",
    "genElectron_pz",
    "genElectron_phi",
    "genElectron_eta",
    "genElectron_energy",
    "genElectron_mass",
    "genElectron_charge",
    "genElectron_parentPDG",
    "n_muons",
    "n_electrons",
    "n_photons",
    "muon_px",
    "muon_py",
    "muon_pz",
    "muon_phi",
    "muon_eta",
    "muon_energy",
    "muon_mass",
    "muon_charge",
    "muon_d0",
    "muon_d0variance",
    "muon_d0signif",
    "muon_z0",
    "muon_z0variance",
    "muon_z0signif",
    "electron_px",
    "electron_py",
    "electron_pz",
    "electron_phi",
    "electron_eta",
    "electron_energy",
    "electron_mass",
    "electron_charge",
    "electron_d0",
    "electron_d0variance",
    "electron_d0signif",
    "electron_z0",
    "electron_z0variance",
    "electron_z0signif",
    "photon_px",
    "photon_py",
    "photon_pz",
    "photon_phi",
    "photon_eta",
    "photon_energy",
    "photon_mass",
    "photon_charge",
    "Emiss_energy",
    "Emiss_p",
    "Emiss_px",
    "Emiss_py",
    "Emiss_pz",
    "Emiss_phi",
    "Emiss_eta",
    "recojet_isG_kt2",
    "recojet_isQ_kt2",
    "recojet_isS_kt2",
    "recojet_isC_kt2",
    "recojet_isB_kt2",
    "jet_nmu_kt2",
    "jet_nel_kt2",
    "jet_nchad_kt2",
    "jet_ngamma_kt2",
    "jet_nnhad_kt2",
    "recojet_isG_kt4",
    "recojet_isQ_kt4",
    "recojet_isS_kt4",
    "recojet_isC_kt4",
    "recojet_isB_kt4",
    "jet_nmu_kt4",
    "jet_nel_kt4",
    "jet_nchad_kt4",
    "jet_ngamma_kt4",
    "jet_nnhad_kt4",
    "recojet_isG_kt6",
    "recojet_isQ_kt6",
    "recojet_isS_kt6",
    "recojet_isC_kt6",
    "recojet_isB_kt6",
    "jet_nmu_kt6",
    "jet_nel_kt6",
    "jet_nchad_kt6",
    "jet_ngamma_kt6",
    "jet_nnhad_kt6",
    "recojet_isG_R5",
    "recojet_isQ_R5",
    "recojet_isS_R5",
    "recojet_isC_R5",
    "recojet_isB_R5",
    "jet_nmu_R5",
    "jet_nel_R5",
    "jet_nchad_R5",
    "jet_ngamma_R5",
    "jet_nnhad_R5",
    "jet_kt2_px",
    "jet_kt2_py",
    "jet_kt2_pz",
    "jet_kt2_phi",
    "jet_kt2_eta",
    "jet_kt2_energy",
    "jet_kt2_mass",
    "jet_kt2_flavor",
    "jet_kt4_px",
    "jet_kt4_py",
    "jet_kt4_pz",
    "jet_kt4_phi",
    "jet_kt4_eta",
    "jet_kt4_energy",
    "jet_kt4_mass",
    "jet_kt4_flavor",
    "jet_kt6_px",
    "jet_kt6_py",
    "jet_kt6_pz",
    "jet_kt6_phi",
    "jet_kt6_eta",
    "jet_kt6_energy",
    "jet_kt6_mass",
    "jet_kt6_flavor",
    "jet_R5_px",
    "jet_R5_py",
    "jet_R5_pz",
    "jet_R5_phi",
    "jet_R5_eta",
    "jet_R5_energy",
    "jet_R5_mass",
    "jet_R5_flavor",
    
]
def flatten_awkward_array(df: pd.DataFrame) -> pd.DataFrame:
    values = np.concatenate(df.values)
    #flat_values = np.concatenate(values)
    return values#flat_values
def concatenate_all_chunks(folder_name: str, chunk_list: list, branch_name: str, flatten: bool) -> pd.DataFrame:
    all_dfs = []
    
    for chunk_name in chunk_list:
            df = load_ntuple_data(folder_name, chunk_name, branch_name)
            if flatten:
                df = flatten_awkward_array(df)
            all_dfs.append(df)
    if(flatten==False):
        answer=pd.concat(all_dfs, ignore_index=True)
        return answer
    if(flatten==True):
        flattened_list = [item for sublist in all_dfs for subsublist in sublist for item in subsublist]
        return flattened_list
    
def make_histogram(df: pd.DataFrame, branch_name: str, bins: int = 50, kde: bool = False):
    fig=plt.figure()
    sns.histplot(df, bins=bins, kde=kde)
    plt.xlabel(branch_name)
    plt.ylabel("Count")
    plt.title(f"Histogram of {branch_name}")
    plt.grid()

from matplotlib.backends.backend_pdf import PdfPages
def save_figs_pdf(figs, pdffile):
    with PdfPages(pdffile) as pdf:
        for fig in figs:
            pdf.savefig(fig)
            plt.close(fig)

def combine_series_to_csv(series_list, filename):
    # Combine the series into a single DataFrame
    combined_df = pd.concat(series_list, axis=1)

    # Save the DataFrame to a CSV file
    combined_df.to_csv(filename, index=False)



n_photons1 = concatenate_all_chunks(folder_name=folder[1], chunk_list=chunk04, branch_name=branch[55], flatten=False)
photon_px1 = concatenate_all_chunks(folder_name=folder[1], chunk_list=chunk04, branch_name=branch[84], flatten=False)
photon_py1 = concatenate_all_chunks(folder_name=folder[1], chunk_list=chunk04, branch_name=branch[85], flatten=False)
photon_pz1 = concatenate_all_chunks(folder_name=folder[1], chunk_list=chunk04, branch_name=branch[86], flatten=False)
photon_phi1 = concatenate_all_chunks(folder_name=folder[1], chunk_list=chunk04, branch_name=branch[87], flatten=False)
photon_eta1 = concatenate_all_chunks(folder_name=folder[1], chunk_list=chunk04, branch_name=branch[88], flatten=False)
photon_energy1 = concatenate_all_chunks(folder_name=folder[1], chunk_list=chunk04, branch_name=branch[89], flatten=False)
photon_mass1 = concatenate_all_chunks(folder_name=folder[1], chunk_list=chunk04, branch_name=branch[90], flatten=False)
photon_charge1 = concatenate_all_chunks(folder_name=folder[1], chunk_list=chunk04, branch_name=branch[91], flatten=False)
photon_pT1 = np.sqrt(photon_px1['photon_px']**2 + photon_py1['photon_py']**2)
photon_deltaD1 = np.sqrt(photon_phi1['photon_phi']**2 + photon_eta1['photon_eta']**2)
photon_p1 = np.sqrt(photon_px1["photon_px"]**2 + photon_py1["photon_py"]**2 + photon_pz1["photon_pz"]**2)
list_photon1 = [photon_pT1, photon_deltaD1, n_photons1[branch[55]], photon_px1[branch[84]], photon_py1[branch[85]], photon_pz1[branch[86]], photon_p1, photon_phi1[branch[87]], photon_eta1[branch[88]], photon_energy1[branch[89]]]
combine_series_to_csv(list_photon1, "photon1.csv")

Emiss_energy1 = concatenate_all_chunks(folder_name=folder[1], chunk_list=chunk04, branch_name=branch[92], flatten=False)
Emiss_p1 = concatenate_all_chunks(folder_name=folder[1], chunk_list=chunk04, branch_name=branch[93], flatten=False)
Emiss_px1 = concatenate_all_chunks(folder_name=folder[1], chunk_list=chunk04, branch_name=branch[94], flatten=False)
Emiss_py1 = concatenate_all_chunks(folder_name=folder[1], chunk_list=chunk04, branch_name=branch[95], flatten=False)
Emiss_pz1 = concatenate_all_chunks(folder_name=folder[1], chunk_list=chunk04, branch_name=branch[96], flatten=False)
Emiss_phi1 = concatenate_all_chunks(folder_name=folder[1], chunk_list=chunk04, branch_name=branch[97], flatten=False)
Emiss_eta1 = concatenate_all_chunks(folder_name=folder[1], chunk_list=chunk04, branch_name=branch[98], flatten=False)
Emiss_pT1 = np.sqrt(Emiss_px1['Emiss_px']**2 + Emiss_py1['Emiss_py']**2)
Emiss_deltaD1 = np.sqrt(Emiss_phi1['Emiss_phi']**2 + Emiss_eta1['Emiss_eta']**2)
list_Emiss1 = [Emiss_pT1, Emiss_deltaD1, Emiss_energy1[branch[92]], Emiss_p1[branch[93]], Emiss_px1[branch[94]], Emiss_py1[branch[95]], Emiss_pz1[branch[96]], Emiss_phi1[branch[97]], Emiss_eta1[branch[98]]]
combine_series_to_csv(list_Emiss1, "Emiss1.csv")

jet_R5_px1 = concatenate_all_chunks(folder[1], chunk04, branch[163], False)
jet_R5_py1 = concatenate_all_chunks(folder[1], chunk04, branch[164], False)
jet_R5_pz1 = concatenate_all_chunks(folder[1], chunk04, branch[165], False)
jet_R5_phi1 = concatenate_all_chunks(folder[1], chunk04, branch[166], False)
jet_R5_eta1 = concatenate_all_chunks(folder[1], chunk04, branch[167], False)
jet_R5_energy1 = concatenate_all_chunks(folder[1], chunk04, branch[168], False)
jet_R5_mass1 = concatenate_all_chunks(folder[1], chunk04, branch[169], False)
jet_R5_flavor1 = concatenate_all_chunks(folder[1], chunk04, branch[170], False)
jet_R5_p1 = np.sqrt(jet_R5_px1['jet_R5_px']**2 + jet_R5_py1['jet_R5_py']**2 + jet_R5_pz1['jet_R5_pz']**2)
jet_R5_pT1 = np.sqrt(jet_R5_px1['jet_R5_px']**2 + jet_R5_py1['jet_R5_py']**2)
jet_R5_deltaD1 = np.sqrt(jet_R5_phi1['jet_R5_phi']**2 + jet_R5_eta1['jet_R5_eta']**2)
list_jet_R51 = [jet_R5_px1[branch[163]], jet_R5_py1[branch[164]], jet_R5_pz1[branch[165]], jet_R5_p1, jet_R5_pT1, jet_R5_phi1[branch[166]], jet_R5_eta1[branch[167]], jet_R5_deltaD1, jet_R5_energy1[branch[168]]]
combine_series_to_csv(list_jet_R51, "jet_R51.csv")

n_photons2 = concatenate_all_chunks(folder_name=folder[2], chunk_list=chunk04, branch_name=branch[55], flatten=False)
photon_px2 = concatenate_all_chunks(folder_name=folder[2], chunk_list=chunk04, branch_name=branch[84], flatten=False)
photon_py2 = concatenate_all_chunks(folder_name=folder[2], chunk_list=chunk04, branch_name=branch[85], flatten=False)
photon_pz2 = concatenate_all_chunks(folder_name=folder[2], chunk_list=chunk04, branch_name=branch[86], flatten=False)
photon_phi2 = concatenate_all_chunks(folder_name=folder[2], chunk_list=chunk04, branch_name=branch[87], flatten=False)
photon_eta2 = concatenate_all_chunks(folder_name=folder[2], chunk_list=chunk04, branch_name=branch[88], flatten=False)
photon_energy2 = concatenate_all_chunks(folder_name=folder[2], chunk_list=chunk04, branch_name=branch[89], flatten=False)
photon_mass2 = concatenate_all_chunks(folder_name=folder[2], chunk_list=chunk04, branch_name=branch[90], flatten=False)
photon_charge2 = concatenate_all_chunks(folder_name=folder[2], chunk_list=chunk04, branch_name=branch[91], flatten=False)
photon_pT2 = np.sqrt(photon_px2['photon_px']**2 + photon_py2['photon_py']**2)
photon_deltaD2 = np.sqrt(photon_phi2['photon_phi']**2 + photon_eta2['photon_eta']**2)
photon_p2 = np.sqrt(photon_px2["photon_px"]**2 + photon_py2["photon_py"]**2 + photon_pz2["photon_pz"]**2)
list_photon2 = [photon_pT2, photon_deltaD2, n_photons2[branch[55]], photon_px2[branch[84]], photon_py2[branch[85]], photon_pz2[branch[86]], photon_p2, photon_phi2[branch[87]], photon_eta2[branch[88]], photon_energy2[branch[89]]]
combine_series_to_csv(list_photon2, "photon2.csv")

Emiss_energy2 = concatenate_all_chunks(folder_name=folder[2], chunk_list=chunk04, branch_name=branch[92], flatten=False)
Emiss_p2 = concatenate_all_chunks(folder_name=folder[2], chunk_list=chunk04, branch_name=branch[93], flatten=False)
Emiss_px2 = concatenate_all_chunks(folder_name=folder[2], chunk_list=chunk04, branch_name=branch[94], flatten=False)
Emiss_py2 = concatenate_all_chunks(folder_name=folder[2], chunk_list=chunk04, branch_name=branch[95], flatten=False)
Emiss_pz2 = concatenate_all_chunks(folder_name=folder[2], chunk_list=chunk04, branch_name=branch[96], flatten=False)
Emiss_phi2 = concatenate_all_chunks(folder_name=folder[2], chunk_list=chunk04, branch_name=branch[97], flatten=False)
Emiss_eta2 = concatenate_all_chunks(folder_name=folder[2], chunk_list=chunk04, branch_name=branch[98], flatten=False)
Emiss_pT2 = np.sqrt(Emiss_px2['Emiss_px']**2 + Emiss_py2['Emiss_py']**2)
Emiss_deltaD2 = np.sqrt(Emiss_phi2['Emiss_phi']**2 + Emiss_eta2['Emiss_eta']**2)
list_Emiss2 = [Emiss_pT2, Emiss_deltaD2, Emiss_energy2[branch[92]], Emiss_p2[branch[93]], Emiss_px2[branch[94]], Emiss_py2[branch[95]], Emiss_pz2[branch[96]], Emiss_phi2[branch[97]], Emiss_eta2[branch[98]]]
combine_series_to_csv(list_Emiss2, "Emiss2.csv")

jet_R5_px2 = concatenate_all_chunks(folder[2], chunk04, branch[163], False)
jet_R5_py2 = concatenate_all_chunks(folder[2], chunk04, branch[164], False)
jet_R5_pz2 = concatenate_all_chunks(folder[2], chunk04, branch[165], False)
jet_R5_phi2 = concatenate_all_chunks(folder[2], chunk04, branch[166], False)
jet_R5_eta2 = concatenate_all_chunks(folder[2], chunk04, branch[167], False)
jet_R5_energy2 = concatenate_all_chunks(folder[2], chunk04, branch[168], False)
jet_R5_mass2 = concatenate_all_chunks(folder[2], chunk04, branch[169], False)
jet_R5_flavor2 = concatenate_all_chunks(folder[2], chunk04, branch[170], False)
jet_R5_p2 = np.sqrt(jet_R5_px2['jet_R5_px']**2 + jet_R5_py2['jet_R5_py']**2 + jet_R5_pz2['jet_R5_pz']**2)
jet_R5_pT2 = np.sqrt(jet_R5_px2['jet_R5_px']**2 + jet_R5_py2['jet_R5_py']**2)
jet_R5_deltaD2 = np.sqrt(jet_R5_phi2['jet_R5_phi']**2 + jet_R5_eta2['jet_R5_eta']**2)
list_jet_R52 = [jet_R5_px2[branch[163]], jet_R5_py2[branch[164]], jet_R5_pz2[branch[165]], jet_R5_p2, jet_R5_pT2, jet_R5_phi2[branch[166]], jet_R5_eta2[branch[167]], jet_R5_deltaD2, jet_R5_energy2[branch[168]]]
combine_series_to_csv(list_jet_R52, "jet_R52.csv")






def filter_particles(csv_file_path, energy_column, energy_threshold = 15, save_path = None):
    df = pd.read_csv(csv_file_path)

    if energy_column not in df.columns:
        raise ValuError(f"The CSV file does not contain a '{energy_column}' column")

    filtered_df = df[df[energy_column] >= energy_threshold]

    if save_path:
        filtered_df.to_csv(save_path, index=False)
        
    return filtered_df


def hist_from_csv(file_path: str, output_path: str, output_file_name: str):
    """
    This function creates a series of histograms from the filtered .csv files
    """
    df = pd.read_csv(file_path)
    column_list = df.columns.tolist()
    # for some reason, sometimes p, pT, and deltaD were renamed into 0, 1, and 2 respectively
    for column in column_list:
        if column == 0: column = 'p'
        elif column == 1: column = 'pT'
        elif column == 2: column = 'deltaD'

    output_pdf_path = os.path.join(output_path, output_file_name)

    with PdfPages(output_pdf_path) as pdf:
        for column in column_list:
            if df[column].dtype in ['int64', 'float64']:
                plt.figure()
                df[column].dropna().plot(kind = 'hist', bins = 50)
                plt.title(f'Histogram of {column}')
                plt.xlabel(column)
                plt.ylabel('frequency')

                pdf.savefig()
                plt.close()

    print("Done")