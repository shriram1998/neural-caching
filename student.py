from adapters.load_mcl import ModularMixin
from train_utils import (
    load_optimizer,
    evaluate_model,
    train_epoch,
    get_hparams,
    get_model,
)
import torch.nn as nn
import torch
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import (
    get_scheduler,
)
from utils import (
    setup_basics,
    EarlyStopper,
    neptune_log,
    set_seeds,
)
import copy
from task import (
    get_task,
)
from metrics import Metric
import pdb
import time
import numpy as np

logger = get_logger(__name__)

LOG_TRAIN = True


class student:
    def __init__(self, args, task, run, accelerator):
        self.cache = []
        self.task_name = args.task_name
        self.budget_arr = [int(elem) for elem in args.budget.split(",")]
        self.seed = args.seed
        self.target = args.target
        self.incremental=args.incremental
        self.learning_rate = args.learning_rate
        self.args = get_hparams(args, self.task_name)
        self.test = task.data["test_dataloader"]
        self.test_wrong = task.data["test_wrong_dataloader"]
        self.run = run
        self.seed = args.seed
        self.accelerator = accelerator
        self.iteration = 0
        self.save_checkpoint = args.save_checkpoint
        self.soft_labels = args.soft_labels
        self.test_scores_gold = [0, 0]
        self.test_scores_llm = [0, 0]
        self.suffixes = [""]
        if task.is_classification:
            self.dic_classes = list(task.classes_dict_gold.values())
        else:
            self.dic_classes = None

        self.metric = Metric(self.args, soft=self.args.is_classification)
        self.metric_test = Metric(self.args, soft=self.args.is_classification)
        self.total_flops=0
        self.total_time_elapsed=0
        self.init_model()

    def init_model(self):
        set_seeds(self.seed)
        model = get_model(self.args)
        self.model = ModularMixin(
            model,
            freeze=True,
            ac_kwargs={
                "r": self.args.r,
                "lora_scaling": self.args.lora_scaling,
                "seed": self.seed,
            },
        )
        return
    
    def init_optimizer_continual(self):
        optimizer = load_optimizer(self.model, self.args, self.learning_rate)

        #linear scheduler: inital lr to zero during training
        #warmup=0 so no warmup phase used
        logger.info(f" Num training steps: {self.args.num_train_epochs * self.budget_arr[-1]}")
        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=int(
                self.args.warmup
                * self.args.num_train_epochs
                * self.budget_arr[-1]
            ),
            num_training_steps=self.args.num_train_epochs * self.budget_arr[-1],
        )
        return optimizer, lr_scheduler

    def init_checkpoint(self, PATH):
        self.model.load_state_dict(torch.load(PATH))
        self.model.cuda()
        return

    def query(self, input):
        '''
        Set the model to evaluation mode.
        Move the model to a GPU for inference.
        Generate predictions based on the input data.
        Handle both soft and hard label predictions.
        '''
        torch.cuda.empty_cache()
        self.model.eval()
        self.model.cuda()
        with torch.no_grad():
            if self.soft_labels:
                predictions = self.model.generate(
                    **{
                        "input_ids": input["input_ids"].cuda(),
                        "attention_mask": input["attention_mask"].cuda(),
                    },
                    max_new_tokens=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
                predictions = [
                    torch.tensor(
                        list(np.array(predictions[1][0].cpu())[0][self.dic_classes])
                    )
                ]
            else:
                predictions = self.model.generate(
                    **{
                        "input_ids": input["input_ids"].cuda(),
                        "attention_mask": input["attention_mask"].cuda(),
                    },
                    num_beams=self.args.num_beams,
                    max_length=self.args.max_out_length,
                    decoder_start_token_id=self.model.model.config.bos_token_id,
                )
        return predictions

    def evaluate(self):
        self.metric_test.reset()
        test_metric_gold = evaluate_model(
            model=self.model,
            accelerator=self.accelerator,
            eval_dataloader=self.test,
            metric=self.metric_test,
            args=self.args,
            dic_classes=self.dic_classes,
            target="gold",
        )

        self.metric_test.reset()
        test_metric_wrong_gold = evaluate_model(
            model=self.model,
            accelerator=self.accelerator,
            eval_dataloader=self.test_wrong,
            metric=self.metric_test,
            args=self.args,
            dic_classes=self.dic_classes,
            target="gold",
        )

        self.metric_test.reset()
        test_metric_llm = evaluate_model(
            model=self.model,
            accelerator=self.accelerator,
            eval_dataloader=self.test,
            metric=self.metric_test,
            args=self.args,
            dic_classes=self.dic_classes,
            target="llm",
        )
        test_metric_wrong_llm = evaluate_model(
            model=self.model,
            accelerator=self.accelerator,
            eval_dataloader=self.test_wrong,
            metric=self.metric_test,
            args=self.args,
            dic_classes=self.dic_classes,
            target="llm",
        )

        if self.run is not None:
            stats = {
                "test_gold_acc": test_metric_gold[0],
                "test_llm_acc": test_metric_llm[0],
                "test_wrong_gold_acc": test_metric_wrong_gold[0],
                "test_wrong_llm_acc": test_metric_wrong_llm[0],
                "data amount": self.data_amount,
            }

            for suffix in self.suffixes:
                neptune_log(
                    run=self.run,
                    pref=f"test/" + suffix,
                    stats=stats,
                    epoch=self.iteration,
                )
        self.test_scores_gold = [self.test_scores_gold[1], test_metric_gold[0]]
        self.test_scores_llm = [self.test_scores_llm[1], test_metric_llm[0]]
        self.suffixes = [""]

    def train(self, train_dataloader, eval_dataloader):
        torch.cuda.empty_cache()
        self.early_stopper = EarlyStopper(self.args.early_stop)
        self.iteration += 1
        if self.seed is not None:
            set_seed(self.args.seed)

        self.metric.reset()

        if self.incremental=="no":
            logger.info(f"  Resetting model, optimizer and scheduler from scratch.")
            # reset model based on argument
            self.init_model()

            # Re-initialise lr_scheduler + optimizer
            optimizer = load_optimizer(self.model, self.args)

            #linear scheduler: inital lr to zero during training
            #warmup=0 so no warmup phase used
            lr_scheduler = get_scheduler(
                name=self.args.lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=int(
                    self.args.warmup
                    * self.args.num_train_epochs
                    * len(
                        train_dataloader.dataset
                    ) 
                ),
                num_training_steps=self.args.num_train_epochs
                * len(
                    train_dataloader.dataset
                ),
            )
        else:
            logger.info(f"  Resetting optimizer and scheduler for Continual learning.")
            optimizer, lr_scheduler = self.init_optimizer_continual()

        logger.info(f"  Running task {self.task_name}")
        logger.info(f"  Num examples = {len(train_dataloader.dataset)}")

        self.data_amount = len(train_dataloader.dataset) + len(eval_dataloader.dataset) 

        # Move to the device
        self.model, optimizer, lr_scheduler = self.accelerator.prepare(
            self.model, optimizer, lr_scheduler
        )

        # for param in self.model.parameters():
        #     logger.info(param.device)
        # for state in self.optimizer.state.values():
        #     for k, v in state.items():
        #         if isinstance(v, torch.Tensor):
        #             logger.info(v.device)

        num_epochs_log=self.args.num_train_epochs
        total_flops_train = 0
        start_time = time.time()

        for epoch in range(0, self.args.num_train_epochs):
            total_loss, total_flops_epoch = train_epoch(
                model=self.model,
                train_dataloader=train_dataloader,
                accelerator=self.accelerator,
                lr_scheduler=lr_scheduler,
                optimizer=optimizer,
                args=self.args,
                dic_classes=self.dic_classes,
            )
            total_flops_train += total_flops_epoch
            self.learning_rate=optimizer.param_groups[0]["lr"]

            if (
                epoch % self.args.eval_every_epochs == 0
                or epoch == self.args.num_train_epochs - 1
            ):
                eval_metrics = evaluate_model(
                    model=self.model,
                    accelerator=self.accelerator,
                    eval_dataloader=eval_dataloader,
                    metric=self.metric,
                    args=self.args,
                    dic_classes=self.dic_classes,
                    target=self.target,
                )
                self.model.cpu()
                self.early_stopper.update(eval_metrics[0], self.model)
                self.model.cuda()

                log_msg = f"    Epoch {epoch} -----> Average_Train_loss: {total_loss / len(train_dataloader.dataset)} ===== Eval_metric: {eval_metrics[0]}"
                logger.info(log_msg)

                if self.run is not None and LOG_TRAIN:
                    self.run[f"{self.iteration}-eval"].log(eval_metrics[0], step=epoch)

            # log metrics are desactivated
            if self.run is not None and LOG_TRAIN:
                stats = {
                    "loss": total_loss / len(train_dataloader.dataset),
                    "main_lr": optimizer.param_groups[0]["lr"],
                }
                logger.info(f"  Loss: {stats['loss']}, LR: {stats['main_lr']}")

            if self.early_stopper.should_finish():
                num_epochs_log=epoch
                break
        
        end_time = time.time()
        elapsed_time = end_time - start_time

        stats={
            "epochs_trained": num_epochs_log,
            "data_amount": self.data_amount,
            "train_examples": len(train_dataloader.dataset),
            "eval_examples": len(eval_dataloader.dataset),
            "flops": total_flops_train,
            "time_elapsed": elapsed_time
        }
        
        logger.info(f"  Total flops for this training: {total_flops_train:.2e}")

        neptune_log(
            run=self.run,
            pref=f"train/",
            stats=stats,
            epoch=self.iteration,
        )

        self.total_flops+=total_flops_train
        self.total_time_elapsed+=elapsed_time
        
        # copying from a cpu
        self.model.cpu()
        self.model = copy.deepcopy(self.early_stopper.get_best())
        self.model = self.early_stopper.get_best().cuda()
        del self.early_stopper.best_model

        self.evaluate()
        if self.save_checkpoint != "no":
            PATH_DEST = (
                "checkpoints/"
                + self.task_name
                + "/"
                + str(self.seed)
                + "_"
                + str(len(train_dataloader.dataset) + len(eval_dataloader.dataset))
                + ".pt"
            )
            torch.save(self.model.state_dict(), PATH_DEST)


class aux_student:
    def __init__(self, model, args, task):
        self.model = model
        self.task_name = args.task_name
        self.args = args
        self.task = task
        self.soft_labels = args.soft_labels
        if task.is_classification:
            self.dic_classes = list(task.classes_dict_gold.values())
        else:
            self.dic_classes = None

    def query(self, input):
        torch.cuda.empty_cache()
        self.model.eval()
        with torch.no_grad():
            if self.soft_labels:
                predictions = self.model.generate(
                    **{
                        "input_ids": input["input_ids"].cpu(), 
                        "attention_mask": input["attention_mask"].cpu(), 
                    },
                    max_new_tokens=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
                predictions = [
                    torch.tensor(
                        list(np.array(predictions[1][0].cpu())[0][self.dic_classes])
                    )
                ]
            else:
                predictions = self.model.generate(
                    **{
                        "input_ids": input["input_ids"], 
                        "attention_mask": input["attention_mask"], 
                    },
                    num_beams=self.args.num_beams,
                    max_length=self.args.max_out_length,
                    decoder_start_token_id=self.model.model.config.bos_token_id,
                )
        self.model.cpu()
        input["input_ids"].cuda()
        input["attention_mask"].cuda()
        return predictions
