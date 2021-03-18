import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

FILE_PATH = './results/inference_times.csv'

START_INDEX = 90
END_INDEX = 104

SAVE_CSV = False
SAVE_IMAGE = True

filename = 'tflite_all'
figure_title = 'Inference speed with TensorFlow Lite'

x_axis_name = 'Model name'
y_axis_name = "Average inference time\n(ms)"
legend_name = 'Device'

x_multiplier = 1000
rotation = 0

show_values_on_top = True
font_size = 18

use_custom_x = True
# custom_x_names = ['MobileNet SSDLite EdgeTPU',
#                     'MobileDet EdgeTPU',
#                     'YOLO v5s  EdgeTPU',
#                     'Tiny-YOLO v4  EdgeTPU']
# custom_x_names = ['TF2 SSD 320',
#                   'TF1 SSDLite 300',
#                   'TF1 SSD 300',
#                   'TF1 SSD quant 300',
#                   'TF2 SSD FPN 320',
#                   'TF1 SSD MnasFPN 320',
#                   'TF1 SSD MnasFPN 640']
custom_x_names = ['MobileDet',
                  'MobileNet SSD',
                  'YOLO v5s',
                  'Tiny-YOLO v4',
                  'EfficientDet D0']
custom_x_names.reverse()
use_custom_legend = use_custom_x
custom_legend_names = ['Raspberry Pi',
                       'Laptop',
                       'Colab GPU backend',
                       'Colab CPU backend',
                       'Desktop',]

df = pd.read_csv(FILE_PATH, squeeze=True)
# df = df.replace('.tflite', '', regex=True)
df = df.loc[
    (df['test_index'] >= START_INDEX) & (df['test_index'] <= END_INDEX)]

result_df = pd.DataFrame(columns=[x_axis_name, legend_name, y_axis_name])

print(df['model_name'].unique())
for model in df['model_name'].unique():
    if 'efficientdetD0' in model or '32' in model in model:
        if 'tf2' not in model and 'edge' not in model and '320' not in model:
            for system in df['system'].unique():
                if 'raspdfg' not in system:

                    locations = df.loc[(df['model_name'] == model)
                                       & (df['system'] == system)]
                    avg_inference = np.average(
                        locations[3::4]['inference_time'].append(
                            locations[2::4]['inference_time']).append(
                            locations[1::4]['inference_time'])) * x_multiplier
                    if system == 'laptop':
                        avg_inference = np.average(
                            locations[4::5]['inference_time'].append(
                                locations[3::5]['inference_time'].append(
                                    locations[2::5]['inference_time']).append(
                                    locations[1::5]['inference_time']))) * x_multiplier
                    result_df = result_df.append(
                        pd.Series([model, system, avg_inference],
                                  index=result_df.columns), ignore_index=True)
result_df = result_df.sort_values(y_axis_name, ascending=False)
# result_df = result_df.groupby('Model name').apply(lambda x: x.sort_values('Average inference time\n(ms)'))
result_df = result_df.reset_index(drop=True)
# result_df = result_df.reindex(list(range(5, 25)) + list(range(0, 5)))
# result_df = result_df.reindex(list(range(24, -1, -1)))

print(result_df)

# result_df.loc[3, y_axis_name] = 100  # If a specific value needs to be changes
plt.close('all')
plt.style.use(['science', 'grid', 'no-latex'])
# plt.style.use('bmh')
plt.figure(figsize=(22, 8))
plt.rcParams.update({'font.size': font_size,
                     'mathtext.fontset': 'stix',
                     'font.family': 'STIXGeneral'})

g = sns.barplot(x=x_axis_name, hue=legend_name, y=y_axis_name, data=result_df)

if figure_title:
    g.set_title(figure_title)

if use_custom_x:
    g.set_xticklabels(custom_x_names)      # Switch to full model names

if use_custom_legend:
    h, l = g.get_legend_handles_labels()
    g.legend(h, custom_legend_names)

g.set_xticklabels(g.get_xticklabels(), rotation=rotation)
plt.xlim(reversed(plt.xlim()))

if show_values_on_top:
    for p in g.patches:  # Show values on top of the bars
        _x, _y = p.get_x() + p.get_width() / 2, p.get_y() + p.get_height() + \
                 g.axes.get_ylim()[1] * 0.007
        value = '{:.0f}'.format(p.get_height())
        g.text(_x, _y, value, ha="center", size=font_size * 0.9)

if SAVE_IMAGE:
    plt.savefig('../../' + filename + '.png', bbox_inches='tight')
    plt.show()
else:
    plt.show()


def increment_filename(path, i=0):
    if os.path.isfile(path):
        path = path.replace('.csv', str(i+1) + '.csv')
        return increment_filename(path, i+1)
    else:
        return path


if SAVE_CSV:
    csv_path = increment_filename('./results/processed_result.csv', i=0)

    result_df.to_csv(csv_path, index=False)
