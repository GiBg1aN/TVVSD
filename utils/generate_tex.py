import pandas as pd
import numpy as np

def main(): 
    columns = ['labels_per_class', 'verb_type', 'representation_type', 'accuracy']
    exp_data = pd.read_csv('generated/experiments.csv', names=columns)

    filtered_data = exp_data[exp_data['labels_per_class'] == 1]
    motions = filtered_data[filtered_data['verb_type'] == 'motions']
    non_motions = filtered_data[filtered_data['verb_type'] == 'non_motions']

    motions_cap = motions[motions['representation_type'] == 'e_caption']['accuracy']
    non_motions_cap = non_motions[non_motions['representation_type'] == 'e_caption']['accuracy']

    motions_obj = motions[motions['representation_type'] == 'e_object']['accuracy']
    non_motions_obj = non_motions[non_motions['representation_type'] == 'e_object']['accuracy']

    motions_comb = motions[motions['representation_type'] == 'e_combined']['accuracy']
    non_motions_comb = non_motions[non_motions['representation_type'] == 'e_combined']['accuracy']

    motions_img = motions[motions['representation_type'] == 'e_image']['accuracy']
    non_motions_img = non_motions[non_motions['representation_type'] == 'e_image']['accuracy']

    motions_img_cap = motions[motions['representation_type'] == 'concat_image_caption']['accuracy']
    non_motions_img_cap = non_motions[non_motions['representation_type'] == 'concat_image_caption']['accuracy']

    motions_img_obj = motions[motions['representation_type'] == 'concat_image_object']['accuracy']
    non_motions_img_obj = non_motions[non_motions['representation_type'] == 'concat_image_object']['accuracy']

    motions_img_text = motions[motions['representation_type'] == 'concat_image_text']['accuracy']
    non_motions_img_text = non_motions[non_motions['representation_type'] == 'concat_image_text']['accuracy']



    table = "\\begin{table*}[t!]\\n\\resizebox{\\textwidth}{!}{\\n\\begin{tabular}{lllllllll}\\n\\hline\\n \\multirow{2}{*}{GOLD} & \\multirow{2}{*}{Verb type} & \\multicolumn{3}{c}{Textual} & \\multirow{2}{*}{VIS (CNN)} & \\multicolumn{3}{c}{Concat (CNN+)} \\\\\\n &  & O & C & Combined (O+C) &  & Object (O) & Captions (C)  & Combined (O+C)  \\\\ \\hline\\n \\multirow{2}{*}{\\begin{tabular}[c]{@{}l@{}}Paper results\\\\ PAMI 2019\\end{tabular}} & Motion & 54,60  & 73,30  & \\textbf{75,60}  & 58,30 & 66,60  & 74,70 & 73,80\\\\\\n & Non-Motion & 57,00  & \\textbf{72,70} & \\textbf{72,60}  & 56,10 & 66,00 & 72,20 & 71,30 \\\\\\n \\hline \\multirow{2}{*}{\\begin{tabular}[c]{@{}l@{}}Semi-supervised\\\\ GTG 1 Label x class\\end{tabular}} & Motion  & \\textbf{%.2f $\\pm$ %.1f} & \\textbf{%.2f $\\pm$ %.1f} & %.2f $\\pm$ %.1f & \\textbf{%.2f $\\pm$ %.1f} & %.2f $\\pm$ %.1f & %.2f $\\pm$ %.1f  & %.2f $\\pm$ %.1f   \\\\\\n & Non-Motion  & \\textbf{%.2f $\\pm$ %.1f} & %.2f $\\pm$ %.1f  & %.2f $\\pm$ %.1f & \\textbf{%.2f $\\pm$ %.1f} & %.2f $\\pm$ %.1f & %.2f $\\pm$ %.1f & %.2f $\\pm$ %.1f \\\\\\n \\hline\\n \\end{tabular}\\n }\\n \\end{table*}" % (motions_obj.mean() * 100, motions_obj.std() * 100, motions_cap.mean() * 100, motions_cap.std() * 100, motions_comb.mean() * 100, motions_comb.std() * 100, motions_img.mean() * 100, motions_img.std() * 100, motions_img_obj.mean() * 100, motions_img_obj.std() * 100, motions_img_cap.mean() * 100, motions_img_cap.std() * 100, motions_img_text.mean() * 100, motions_img_text.std() * 100, non_motions_obj.mean() * 100, non_motions_obj.std() * 100, non_motions_cap.mean() * 100, non_motions_cap.std() * 100, non_motions_comb.mean() * 100, non_motions_comb.std() * 100, non_motions_img.mean() * 100, non_motions_img.std() * 100, non_motions_img_obj.mean() * 100, non_motions_img_obj.std() * 100, non_motions_img_cap.mean() * 100, non_motions_img_cap.std() * 100, non_motions_img_text.mean() * 100, non_motions_img_text.std() * 100) 

    return table
