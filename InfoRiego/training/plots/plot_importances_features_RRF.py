import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns


def plot_importances_features_RRF():
    """ This method creates a seperate figure for every prediction horizon
        Plots importance of single most important feature per station """
    
    df_features = pd.read_csv('df_importances_RRF.csv') 
    horizon = [0.5,1,1.5,2,3,4]
    for i in range (len(horizon)):
        fig = plt.figure(i,figsize=(12,9))
        plt.style.use('seaborn-whitegrid')
        castilla_and_leon_img=mpimg.imread('Castilla_and_Leon_Provinces.png')
        ax = fig.add_subplot(111)
        pic = df_features.plot.scatter(x= "WO_dist", y="NS_dist",ax=ax,c=df_features[df_features.columns[11+i]],cmap='brg',
                colorbar= False,s=df_features[df_features.columns[17+i]]*3500,label = 'size: feature importance',alpha=1)
        ax.imshow(castilla_and_leon_img, extent=[-140, 290,-240, 118], alpha=0.5)
        ax.set_xlabel('X [km]',fontsize=20)
        ax.set_ylabel('Y [km]', fontsize=20)            
        ax.set_title('Random Forest - prediction horizon: {0} hour(s)'.format(horizon[i]),fontsize=20)            
        ax.annotate('target', xy=(0,7), xytext=(-5,70),arrowprops=dict(arrowstyle="->"),fontsize=20,alpha=1)        
        ax.annotate('radiation T3',xy=(150,-165), xytext=(150,-165),color ='sienna',fontsize=18)
        ax.annotate('radiation T2',xy=(150,-150), xytext=(150,-150),color ='mediumvioletred',fontsize=18)  
        ax.annotate('azimuth T3',xy=(150,-195), xytext=(150,-195),color ='lime',fontsize=18)
        ax.annotate('radiation T1',xy=(150,-135), xytext=(150,-135),color ='blue',fontsize=18)
        leg =ax.legend(loc=4,fancybox=True,prop={'size': 13},frameon=True,framealpha=1)
        leg.legendHandles[0].set_color('gray')

    plt.show()

plot_importances_features_RRF()
