import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns


def plot_importances_stations_ARX():
    """ This method creates a seperate figure for every prediction horizon
        Plots importance of summe of all features per station """
    
    df_features = pd.read_csv('df_importances_ARX.csv') 
    horizon = [0.5,1,1.5,2,3,4]
    for i in range (len(horizon)):
        fig = plt.figure(i,figsize=(12,9))
        plt.style.use('seaborn-whitegrid')
        castilla_and_leon_img=mpimg.imread('Castilla_and_Leon_Provinces.png')
        ax = fig.add_subplot(111)
        pic = df_features.plot.scatter(x= "WO_dist", y="NS_dist",ax=ax, c=df_features[df_features.columns[5+i]],cmap='gnuplot',colorbar=True,s=100,alpha=1)
        ax.imshow(castilla_and_leon_img, extent=[-140, 290,-240, 118], alpha=0.5)
        ax.set_xlabel('X [km]',fontsize=20)
        ax.set_ylabel('Y [km]', fontsize=20)            
        ax.set_title('Linear Regression - prediction horizon: {0} hour(s)'.format(horizon[i]),fontsize=20)            
        ax.annotate('target', xy=(0,7), xytext=(-5,70),arrowprops=dict(arrowstyle="->"),fontsize=20,alpha=1)        
        
    plt.show()

plot_importances_stations_ARX()
