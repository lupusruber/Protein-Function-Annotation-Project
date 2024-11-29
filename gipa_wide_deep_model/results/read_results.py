import torch
from sklearn.metrics import classification_report, roc_auc_score

for ont in ('BP', 'MF', 'CC'):

    for t in (700, 900):
        true = torch.load(f'/home/lupusruber/root/projects/ppi/PPI/gipa_wide_deep_model/results/whole_graph_data_{ont}_{t}_labels.pt').numpy()
        pred = torch.load(f'/home/lupusruber/root/projects/ppi/PPI/gipa_wide_deep_model/results/whole_graph_data_{ont}_{t}_preds.pt').numpy()
        
        test_score = roc_auc_score(y_true=true, y_score=pred)

        report = classification_report(
            y_true=true,
            y_pred=pred,
            output_dict=True,
             zero_division=0
        ) | {"auc_score": test_score}

        dataset_name = f'whole_graph_data_{ont}_{t}'
        #with open(f"/home/lupusruber/root/projects/ppi/PPI/gipa_wide_deep_model/results/new_results/{dataset_name}_results.json", "w") as file:
            #file.write(json.dumps(report))
        sample_avg = report['samples avg']
        # & 1 & 1 & 1 & 1 
        # f1 prec rec auc
        print('='*40)
        print(dataset_name)
        string_to_print = f"& {sample_avg['f1-score']:.4f} & {sample_avg['precision']:.4f} & {sample_avg['recall']:.4f} & {report['auc_score']:.4f}"
        print(string_to_print)
        print('='*40)
