from Bio.SeqUtils.ProtParam import ProteinAnalysis
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np

def get_properties(input_data):
    if isinstance(input_data, str):
        if input_data.isalpha():
            # 处理蛋白质序列
            protein = ProteinAnalysis(input_data)
            molecular_weight = protein.molecular_weight()
            iso_point = protein.isoelectric_point()
            secondary_structure_fraction = protein.secondary_structure_fraction()
            gravy = protein.gravy()
            sasa = protein.molar_extinction_coefficient()
            #print(f"分子量: {molecular_weight} Da")
            #print(f"等电点 (pI): {iso_point}")
            #print(f"二级结构比例 (α螺旋, β折叠, 线圈): {secondary_structure_fraction}")
            #print(f"疏水性 (GRAVY): {gravy}")
            #print(f"表面可及面积 (SASA): {sasa}")
            # 构造物化性质列表
            properties = [
                molecular_weight,
                iso_point,
                secondary_structure_fraction[0],  # α螺旋比例
                secondary_structure_fraction[1],  # β折叠比例
                secondary_structure_fraction[2],  # 线圈比例
                gravy,
                sum(sasa)  # 将表面可及面积的两个值求和表示总的SASA
            ]
            
        else:
            # 处理SMILES字符串
            molecule = Chem.MolFromSmiles(input_data)
            mol_weight = Descriptors.MolWt(molecule)
            logp = Descriptors.MolLogP(molecule)
            tpsa = Descriptors.TPSA(molecule)
            num_h_donors = Descriptors.NumHDonors(molecule)
            num_h_acceptors = Descriptors.NumHAcceptors(molecule)
            molecular_refractivity = Descriptors.MolMR(molecule)
            num_rotatable_bonds = Descriptors.NumRotatableBonds(molecule)
            #print(f"分子量: {mol_weight} g/mol")
            #print(f"LogP: {logp}")
            #print(f"极性表面积 (PSA): {tpsa} Å²")
            #print(f"氢键供体数量: {num_h_donors}")
            #print(f"氢键受体数量: {num_h_acceptors}")
            #print(f"分子折射率: {molecular_refractivity}")
            #print(f"旋转键数量: {num_rotatable_bonds}")
            # 构造物化性质列表
            properties = [
                mol_weight,
                logp,
                tpsa,
                num_h_donors,
                num_h_acceptors,
                molecular_refractivity,
                num_rotatable_bonds
            ]
    else:
        raise ValueError("输入数据类型错误，应为蛋白质序列字符串或SMILES字符串")
    
    return np.array(properties)