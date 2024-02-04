import os
from clip_finetune.losses.flyp_loss import flyp_loss
from clip_finetune.models.modeling import ClassificationHead, CLIPEncoder
from clip_finetune.args import parse_arguments
import logging


def main(args):

    ###logging##################################################################
    os.makedirs(args.save_dir + args.exp_name, exist_ok=True)
    args.save_dir = args.save_dir + args.exp_name + "/" + "_BS" + str(
        args.batch_size) + "_WD" + str(args.weight_decay) + "_LR" + str(args.lr) + "_run" + str(args.run)
    os.makedirs("expt_logs/" + args.exp_name, exist_ok=True)
    logging_path = "expt_logs/" + args.exp_name + "/" + "_BS" + str(
        args.batch_size) + "_WD" + str(args.weight_decay) + "_LR" + str(args.lr) + "_run" + str(args.run)
    os.makedirs(logging_path, exist_ok=True)
    log_filename = logging_path + "/log.log"
    logging.basicConfig(filename=log_filename,
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    assert args.save_dir is not None, 'Please provide a path to store models'
    #############################################################################

    # Initialize the CLIP encoder
    clip_encoder = CLIPEncoder(args, keep_text_modules=True)
    classification_head = ClassificationHead(normalize_input=True, weight=None, use_bias=args.clf_bias,
                                             normalize_weight=args.norm_clf_weight)
    logger.info(args)
    finetuned_checkpoint = flyp_loss(args, clip_encoder,
                                           classification_head, logger)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
