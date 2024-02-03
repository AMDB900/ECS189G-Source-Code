"""
Concrete SettingModule class for a specific experimental SettingModule
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
import torch


class Setting_Train_Test(setting):
    def load_run_save_evaluate(self):
        # load dataset
        loaded_data = self.dataset.load()

        # run MethodModule
        self.method.data = loaded_data
        train_result, test_result, loss_history = self.method.run()

        # save raw ResultModule
        self.result.data = test_result
        self.result.save()

        self.evaluate.data = train_result
        train_evaluations = self.evaluate.evaluate()

        self.evaluate.data = test_result
        test_evaluations = self.evaluate.evaluate()

        return train_evaluations, test_evaluations, loss_history
