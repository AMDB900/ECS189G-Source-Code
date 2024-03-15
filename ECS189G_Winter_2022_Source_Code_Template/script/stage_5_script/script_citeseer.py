from code.stage_5_code.Dataset_Loader import Dataset_Loader
from code.stage_5_code.Method_Citeseer import Method_Citeseer
from code.stage_2_code.Result_Saver import Result_Saver
from code.stage_2_code.Setting_Train_Test import Setting_Train_Test
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch
import matplotlib.pyplot as plt

# run with
# python3 -m script.stage_5_script.script_citeseer

# ---- Multi-Layer Perceptron script ----
statsMap = {}
g_max_epoch = 350
g_dropout = 0.95
g_weight_decay = 0.187
comboList = [
    # [350, 5e-4, 500, 0.95, 0.187],
    [5e-4, 500],
    [5.1e-4, 500],
    [5.2e-4, 500],
]
if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # ------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader("train", "")
    data_obj.dataset_source_folder_path = "data/stage_5_data/citeseer"
    data_obj.dataset_name = "citeseer"

    for combo in comboList:
        # max_epoch = combo[0]
        # learning_rate = combo[1]
        # hidden_size = combo[2]
        # dropout = combo[3]
        # weight_decay = combo[4]

        max_epoch = g_max_epoch
        learning_rate = combo[0]
        hidden_size = combo[1]
        dropout = g_dropout
        weight_decay = g_weight_decay

        method_obj = Method_Citeseer("GCN ", "",
                                     max_epoch,
                                     learning_rate,
                                     hidden_size,
                                     dropout,
                                     weight_decay
                                     ).to(device)

        result_obj = Result_Saver("saver", "")
        result_obj.result_destination_folder_path = "result/stage_5_result/citeseer_"
        result_obj.result_destination_file_name = "prediction_result"

        setting_obj = Setting_Train_Test("pre split", "")

        evaluate_obj = Evaluate_Accuracy("accuracy", "")
        # ------------------------------------------------------

        # ---- running section ---------------------------------
        data_name = "Citeseer"
        print("************ Start ************")
        setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
        setting_obj.print_setup_summary()
        (
            train_evaluations,
            test_evaluations,
            accuracy_history,
        ) = setting_obj.load_run_save_evaluate()
        accuracy, precision, recall, fscore = train_evaluations
        print("************ Training Set Performance ************")
        print(data_name + " Accuracy: " + str(accuracy))
        print(data_name + " Precision: " + str(precision))
        print(data_name + " Recall: " + str(recall))
        print(data_name + " FScore: " + str(fscore))
        accuracy, precision, recall, fscore = test_evaluations
        print("************ Testing Set Performance ************")
        print(data_name + " Accuracy: " + str(accuracy))

        statsMap[[max_epoch, learning_rate, hidden_size, dropout, weight_decay]] = accuracy

        print(data_name + " Precision: " + str(precision))
        print(data_name + " Recall: " + str(recall))
        print(data_name + " FScore: " + str(fscore))
        print("************ Finish ************")

        plt.clf()
        plt.plot(range(len(accuracy_history)), accuracy_history)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(data_name + " Loss over Epochs")
        plt.savefig(result_obj.result_destination_folder_path + "learning_graph.png")
    # ------------------------------------------------------

    print(statsMap)
    # to rank each combination by accuracy
    # dict(sorted(statsMap.items(), key=lambda item: item[1]))