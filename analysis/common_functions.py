# For Time Logging
from contextlib import contextmanager
import time

import logging

@contextmanager
# Timing Function
def time_usage(name=""):
    """
    log the time usage in a code block
    """
    #print ("In time_usage runID = {}".format(runID))
    start = time.time()
    yield
    end = time.time()
    elapsed_seconds = float("%.10f" % (end - start))
    logging.info('%s: Time Taken (seconds): %s', name, elapsed_seconds)

def train_classifier(estimator,X,y,cv,scl_obj=None,printROC=False,verbose=False):
    """
    # Manual training of SVM using loops (does not parallelize)
    # Not recommended for detailed analysis (use Scikit Learn's internal functions with njobs for parallelism)
    # Only useful whe you want to vizualize the confusion matrix and the classification report
    """
    
    # For ROC
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    for iter_num, (train_indices, test_indices) in enumerate(cv.split(X,y)):
        with time_usage(" SVM: Iteration " + str(iter_num+1)):
            X_train = X[train_indices]
            y_train = y[train_indices]

            X_test = X[test_indices]
            y_test = y[test_indices]

            if (scl_obj == None):
                scl_obj = StandardScaler(with_mean=False,with_std=False)
                
            scl_obj.fit(X_train) 
            X_train_scaled = scl_obj.transform(X_train) # apply to training
            X_test_scaled = scl_obj.transform(X_test) # apply those means and std to the test set (without snooping at the test set values)

            estimator.fit(X_train_scaled,y_train)  # train object

            y_hat = estimator.predict(X_test_scaled) # get test set precitions
            
                        
            if (printROC == True):
                print_classification_details(actual=y_test, predicted=y_hat,verbose=False)
            else:
                print_classification_details(actual=y_test, predicted=y_hat,verbose=verbose)
            
            if (printROC == True):    
                ## For ROC
                probas_ = estimator.predict_proba(X_test_scaled)
                # Compute ROC curve and area the curve
                fpr, tpr, thresholds = mt.roc_curve(y_test, probas_[:, 1])
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                roc_auc = mt.auc(fpr, tpr)
                aucs.append(roc_auc)
                plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (iter_num, roc_auc))
            

    if (printROC == True):
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = mt.auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.legend(loc="lower right")
        plt.show()
		
def print_classification_details(clf, X_test, y_test, display_labels=None, normalize=False, verbose=False):
    # print the accuracy and confusion matrix 
	y_hat = piped_object.predict(X_test)
	cm = mt.confusion_matrix(y_test, y_hat)
	cr = mt.classification_report(y_test, y_hat)

	# print("confusion matrix\n", cm)
	plot_confusion_matrix_wrapper(clf, X_test, y_test, display_labels=display_labels, normalize=normalize)
	print(cr)
	
	if (verbose == True):
		plot_classification_report(cr)
    
def plot_confusion_matrix_wrapper(clf, X_test, y_test, display_labels=None, normalize=False):
	# Plot non-normalized confusion matrix
	titles_options = [("Confusion matrix, without normalization", None)]
	
	if normalize:
		titles_options.append(("Normalized confusion matrix", 'true'))
		
	print(titles_options)
	
	for title, normalize in titles_options:
		disp = plot_confusion_matrix(clf, X_test, y_test,
									 display_labels=display_labels,
									 normalize=normalize)
		disp.ax_.set_title(title)

		print(title)
		#print(disp.confusion_matrix)

	plt.show()
	
	
def plot_classification_report(cr, title=None, cmap='viridis'):
    """
    Adapted from https://medium.com/district-data-labs/visual-diagnostics-for-more-informed-machine-learning-7ec92960c96b
    """
    title = title or 'Classification report'
    lines = cr.split('\n')
    classes = []
    matrix = []
  
    for line in lines[2:(len(lines)-5)]:
        s = line.split()
        classes.append(s[0])
        value = [float(x) for x in s[1: len(s) - 1]]
        matrix.append(value)

    fig, ax = plt.subplots(1)

    for column in range(len(matrix)+1):
        for row in range(len(classes)):
            txt = matrix[row][column]
            #ax.text(column,row,matrix[row][column],va='center',ha='center')
            ax.text(column,row,txt,va='center',ha='center',size="x-large",bbox=dict(facecolor='white', alpha=0.5))

    fig = plt.imshow(matrix, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    plt.title(title)
    plt.colorbar()
    x_tick_marks = np.arange(len(classes)+1)
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, ['Precision', 'Recall', 'F1-score'], rotation=45)
    plt.yticks(y_tick_marks, classes)
    plt.ylabel('Classes')
    plt.xlabel('Measures')
    plt.show()
	
	
def varImpSVM(model,top,colNames,theme='',figsize=[10,10],classId=0):
    #based on https://medium.com/@aneesha/visualising-top-features-in-linear-svm-with-scikit-learn-and-matplotlib-3454ab18a14d
#    model.fit(X,y)
    importances =  pd.DataFrame(data={'value':np.squeeze(model.coef_[[classId]])
                                ,'name':colNames
                            })
    
    if(theme!=''):
        selImportances = importances.loc[importances['name'].isin(theme)]
    else:
        selImportances = importances
    if(top>selImportances.shape[0]): top=selImportances.shape[0]-1
    
    selImportances = selImportances.iloc[(-selImportances['value'].abs()).argsort()][0:top].sort_values('value',ascending=False)
   # print(selImportances[list(['name','value'])].to_string(index=False))
    colors = ['blue' if c > 0 else 'red' for c in selImportances['value']]
    p1=selImportances.plot.barh(y='value',x='name',figsize=figsize,color=colors)
    plt.title("Features Importance")

    for p in p1.patches:
        p1.annotate(str(round(p.get_width(),4)),(p.get_width(),p.get_y()+.45))
    plt.gca().invert_yaxis()
    plt.show()
	
def varImpRFStDev(model,maxCols,colNames):
    
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_],axis=0)
    indices = np.argsort(importances)[::-1][0:maxCols-1]

    #for f in range(maxCols-1): print("%d. feature %s (%f)" % (f + 1, colNames[f], importances[indices[f]]))
    plt.figure(figsize=[10,10])
    plt.title("Features Importance")
    #plt.bar(range(X.shape[1]), importances[indices],color="r", yerr=std[indices], align="center")
    plt.barh(range(maxCols-1), importances[indices],color="r", xerr=std[indices], align="center")
    plt.yticks(range(maxCols-1), colNames[indices])
    plt.ylim([-1, maxCols])
    for i, v in enumerate(importances[indices]):
        plt.text(v, i , str(round(v,4)), color='blue')
    plt.gca().invert_yaxis()
    plt.show()
