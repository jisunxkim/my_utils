import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.backends.backend_pdf
import pandas as pd


from sklearn.metrics import classification_report, roc_curve, roc_auc_score, confusion_matrix
from sklearn.preprocessing import normalize


def plot_hr_ndcg(model_metrics, cols_to_rank, pdf_out=False):
    """
    parameters:
        hr_ndcg_metrics: output from get_hr_ndcg(), heat ratio and NDCG scores
    """
    sns.set(rc={'figure.figsize':(21,10), 'lines.linewidth': 1.5, 'lines.markersize': 20})
    sns.set(style="darkgrid", font_scale=1.2)
    
    top_k = model_metrics.Ranks.max()
    
    if pdf_out:
        time_log = datetime.now().strftime('%Y%m%d_%H:%M')
        pdf_file_name = f"./tmp/hr_ndcg_{time_log}.pdf"
        pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_file_name)

    fig, axes = plt.subplots(2,1)
    sns.lineplot(
        data=model_metrics, 
        x='Ranks', y='hit_rate', 
        ax=axes[0],
        marker='o',
        hue='Treatment'
    )

    axes[0].set_xlabel("Rank K")
    axes[0].set_ylabel("HR@K")
    axes[0].legend(frameon=True)

    sns.lineplot(
        data=model_metrics, 
        x='Ranks', y='ndcg', 
        ax=axes[1],
        marker='o',
        hue='Treatment'
    )

    axes[1].set_ylabel("NDCG@K")
    axes[1].set_xlabel("Rank K")
    axes[1].legend(frameon=True)

    fig.suptitle(f"Hit Rate and NDCG of TOP {top_k} Ranks by {cols_to_rank}", size=20)
    
    if pdf_out:
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        pdf.close()
        print(f"Saved charts as a pdf, {pdf_file_name}")
    else:
        plt.tight_layout()
        plt.show()
    


def plot_feature_importance(sorted_feature_importance, model_name="", font_size = 10, save_fig = False):
    # Extract the feature names and importance values
    feature_names = list(sorted_feature_importance.keys())
    importance_values = list(sorted_feature_importance.values())
    
    fontsize = font_size  # Set the desired font size for y-axis labels
    figure_height = len(feature_names) * fontsize * 0.02  # Adjust this factor to your preference


    # Create a bar chart
    plt.figure(figsize=(10, figure_height)) # increase the size to give more space between labels
    plt.barh(range(len(feature_names)), importance_values, tick_label=feature_names)

    # Add labels and title
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.title(f'Feature Importance: {model_name}')
    
    # Customize the figure
    plt.yticks(range(len(feature_names)), feature_names, fontsize=fontsize)  # Set font size for labels
    plt.gca().invert_yaxis() # Invert the y-axis to display the features in descending order
    
    # Adjust margins to reduce empty space at the top and bottom
    plt.margins(y=0, tight=True)
    
    # Save the plot
    if save_fig: 
        plt.savefig(f'./tmp/feature_importance_plot_{model_name}.png', bbox_inches='tight')

    # Show the plot
    plt.tight_layout()
    plt.show()
    

# Create confusion matrix
def plot_confusion_matrix (
    true_labels, pred_labels,
    title = None,
    target_names = ['class 0', 'class 1'],
    print_cm = False,
    display_image = True,
    save_image = False
):

    cm = confusion_matrix(true_labels, pred_labels)
    cm_df = pd.DataFrame(cm, index = target_names, columns = target_names)
    cm_normalize_df = pd.DataFrame(normalize(cm, 'l1', axis = 1), index = target_names, columns = target_names)
    
    if print_cm:
        print(cm_normalize_df)

    if display_image:
        fig, axes = plt.subplots(1, 2,
                                 figsize = (8,4),
                                 constrained_layout = True)
        
        svm = sns.heatmap(cm_df, annot=True,cmap=plt.cm.Blues, ax=axes[0])
        # plt.tight_layout()
        axes[0].set_ylabel('True label (counts)')
        axes[0].set_xlabel('Predicted label (counts)')

        # plt.subplot(1, 2, 2, inlayer=True)
        svm = sns.heatmap(cm_normalize_df, annot=True,cmap=plt.cm.Blues, ax=axes[1])
        axes[1].set_ylabel('True label (normalized)')
        axes[1].set_xlabel('Predicted label (normalized)')
        
        fig.suptitle(title)
        
        plt.show()

    if display_image and save_image:
        plt.savefig("cm.png")