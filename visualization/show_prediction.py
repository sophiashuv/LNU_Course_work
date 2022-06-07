import matplotlib.patches as mpatches
import sys
sys.path.insert(0, '..')

from model_predict import *
from metrics import *


def plot_confusion_matrix(y_true, y_pred):
    alphabets = ['1', '0']
    TN, FP, FN, TP = get_confusion_matrix(y_true.ravel(), y_pred.ravel())
    arr = (np.array([[TP, FP], [FN, TN]])/(TN + FP + FN + TP))*100

    figure = plt.figure()
    axes = figure.add_subplot(111)

    caxes = axes.matshow(arr, interpolation='nearest', cmap=plt.get_cmap("summer"))
    figure.colorbar(caxes)
    for (i, j), z in np.ndenumerate(arr):
        axes.text(j, i, '{:0.1f}%'.format(z), ha='center', va='center')

    axes.set_xticklabels([''] + alphabets)
    axes.set_yticklabels([''] + alphabets)
    axes.set_xlabel('Predicted label')
    axes.set_ylabel('True label')
    plt.title("Confusion matrix (U-Net)")
    plt.show()


def show_image_prediction(architecture, test_ids, X_test, Y_test, Y_pred):
    plot1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    plt.suptitle(architecture + " segmentation of " + test_ids[0] + "\n", fontsize="x-large")
    ax1.imshow(X_test[0])
    ax2.imshow(np.squeeze(Y_test[0]))
    im = ax3.imshow(np.squeeze(Y_pred[0]))
    plt.colorbar(im, ax=(ax1, ax2, ax3))
    ax1.set_title("X_test", fontsize="x-large")
    ax2.set_title("Y_test", fontsize="x-large")
    ax3.set_title("Y_predicted", fontsize="x-large")
    plt.show()


def show_confusion_matrix_img(architecture, test_ids, X_test, Y_test, Y_pred):
    sum = Y_pred[0] + Y_test[0]
    differ = Y_pred[0] - Y_test[0]
    img = X_test[0]
    color = np.zeros((img.shape), dtype=np.uint8)
    color[differ == 1] = np.array([219, 62, 50])
    color[differ == -1] = np.array([83, 161, 67])
    color[sum == 2] = np.array([31, 117, 204])

    img[sum != 0] = cv2.addWeighted(img[sum != 0], 0.6, color[sum != 0], 0.4, 0)

    plt.figure(figsize=(18, 9))
    plt.imshow(img)
    patches = [mpatches.Patch(color="#db3e32", label="FALSE POSITIVE"),
               mpatches.Patch(color="#53a143", label="FALSE NEGATIVE"),
               mpatches.Patch(color="#1f75cc", label="TRUE POSITIVE")]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title(architecture + "\nConfusion matrix for " + test_ids[0])
    plt.show()


def show_intersections_on_img(architecture, test_ids, X_test, Y_test, Y_pred):
    sum = Y_pred[0] + Y_test[0]
    differ = Y_pred[0] - Y_test[0]
    img = X_test[0]
    color = np.zeros((img.shape), dtype=np.uint8)
    # color[differ == 1] = np.array([219, 62, 50])
    color[sum == 1] = np.array([219, 62, 50])
    color[sum == 2] = np.array([83, 161, 67])

    img[sum != 0] = cv2.addWeighted(img[sum != 0], 0.5, color[sum != 0], 0.5, 0)

    plt.figure(figsize=(18, 9))
    plt.imshow(img)
    patches = [mpatches.Patch(color="#db3e32", label="Difference"),
               mpatches.Patch(color="#53a143", label="Intersection")]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize="x-large")
    plt.title(architecture + "\nIntersection and Difference of Ground Truth and Prediction for \n" + test_ids[0] + "\n", fontsize="x-large")
    plt.show()


def show_plot(architecture):
    if architecture == 'U-Net':
        model = U_Net(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
    elif architecture == 'FCN-8':
        model = FCN(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS, fcn8=True)
    elif architecture == 'FCN-16':
        model = FCN(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS, fcn16=True)
    elif architecture == 'FCN-32':
        model = FCN(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
    else:
        return

    # if way == '1':
    #     test_ids, X_test, Y_test, Y_pred = predict_image(model, IMG_PATH, TEST_PATH, MASK_TEST_PATH, WEIGHTS_PATH, THRESHOLD)
    # elif way == '2':
    #     test_ids, X_test, Y_test, Y_pred = read_predicted_image(IMG_PATH, TEST_PATH, MASK_TEST_PATH, PRED_PATH, THRESHOLD, FORMAT)
    # else:
    #     return
    if task == '1':
        test_ids, X_test, Y_test, Y_pred = predict_image(model, IMG_PATH, TEST_PATH, MASK_TEST_PATH, WEIGHTS_PATH,
                                                         'no')
        show_image_prediction(architecture, test_ids, X_test, Y_test, Y_pred)
    if task == '2':
        test_ids, X_test, Y_test, Y_pred = predict_image(model, IMG_PATH, TEST_PATH, MASK_TEST_PATH, WEIGHTS_PATH,
                                                         THRESHOLD)
        show_confusion_matrix_img(architecture, test_ids, X_test, Y_test, Y_pred)
    if task == '3':
        test_ids, X_test, Y_test, Y_pred = predict_image(model, IMG_PATH, TEST_PATH, MASK_TEST_PATH, WEIGHTS_PATH,
                                                         THRESHOLD)
        show_intersections_on_img(architecture, test_ids, X_test, Y_test, Y_pred)
    if task == '4':
        test_ids, X_test, Y_test, Y_pred = predict_images(model, TEST_PATH, MASK_TEST_PATH, WEIGHTS_PATH, THRESHOLD)
        plot_confusion_matrix(Y_test, Y_pred)


if __name__ == '__main__':
    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    IMG_CHANNELS = 3

    parser = argparse.ArgumentParser()
    parser.add_argument("--task",
                        default='4',
                        help='1 - show image prediction, 2 - show confusion matrix on prediction, '
                             '3 - show intersection and difference, 4 - show confusion matrix')
    parser.add_argument("--way",
                        default='1',
                        help='1 - predict, 2 - read predictions')
    parser.add_argument('--architecture',
                        default='FCN-32',
                        help='Possible: U-Net, FCN-8, FCN-16, FCN-32')
    parser.add_argument('--THRESHOLD',
                        default=0.7)
    parser.add_argument("--TEST_PATH",
                        default=r'C:\Users\sophi\OneDrive\Desktop\inherited_dataset\images\2018_01_08_tes/',
                        help='TEST PATH')
    parser.add_argument("--MASK_TEST_PATH",
                        default=r'C:\Users\sophi\OneDrive\Desktop\inherited_dataset\masks\2018_01_08_tes/',
                        help='TEST MASKS PATH')
    parser.add_argument("--PRED_PATH",
                        default=r'D:\Azure Repository\LNU_Course_work\predictions\U-Net/',
                        help='PRED_PATH')
    parser.add_argument("--FORMAT",
                        default=r'.npy',
                        help='Possible: .npy, .jpg')
    parser.add_argument("--WEIGHTS_PATH",
                        default=r"D:\Azure Repository\LNU_Course_work\FCN32_data\FCN-32_model_epoch=7_valloss=0.0885.h5")
    parser.add_argument("--IMG_PATH",
                        default=r"A5405A44-70FA-4FC5-880C-A5276D411BDF.jpg")
    # E3D71D0F-B27E-45D1-8C8A-95490A6E5400.jpg
    args = parser.parse_args()
    task = args.task
    way = args.way
    architecture = args.architecture
    THRESHOLD = args.THRESHOLD
    TEST_PATH = args.TEST_PATH
    MASK_TEST_PATH = args.MASK_TEST_PATH
    PRED_PATH = args.PRED_PATH
    FORMAT = args.FORMAT
    WEIGHTS_PATH = args.WEIGHTS_PATH
    IMG_PATH = args.IMG_PATH

    show_plot(architecture)
