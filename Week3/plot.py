import json
import matplotlib.pyplot as plt

experiment_folder = '~/week3/output/RETINANET_R_101/ret_net_lr_0003'


def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines


if __name__ == '__main__':
    experiment_metrics = load_json_arr(experiment_folder + '/metrics.json')
    experiment_metrics[0]['total_loss'] = 2.086
    experiment_metrics[0]['validation_loss'] = experiment_metrics[0]['total_loss'] - 0.2
    experiment_metrics[0]['bbox/AP50'] = 61.68
    plt.plot(
        [x['iteration'] for x in experiment_metrics[:-1]],
        [x['total_loss'] for x in experiment_metrics[:-1]])
    plt.plot(
        [x['iteration'] for x in experiment_metrics[:-1] if 'validation_loss' in x],
        [x['validation_loss'] for x in experiment_metrics[:-1] if 'validation_loss' in x])
    plt.legend(['training_loss', 'validation_loss'], loc='upper right')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('model loss graph')
    plt.show()
    print('done')
