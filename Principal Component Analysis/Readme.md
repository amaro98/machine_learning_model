<h1>Principal Component Analysis</h1>

<h2>Overview</h2>
<p>
  Principal Component Analysis is a part of singular value decomposition (SVD). The SVD plays an important role in data compression and finding hidden features. PCA works by taking input and return into a new reconstructed dataset. PCA in this context are used to identify hidden patterns from high dimension data (or to out simple, data with numerous features). Other uses of PCA is for dimensionality reduction
</p>
<p>
  My application for PCA is to identify patterns from a dataset with 60++ features. I want to find at what time where a process change/transition occurs
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/88897287/138472322-4f798d4c-9ff7-427d-999c-a76d836ebc50.png">
</p>

<p> The oversimpified illustration on how is this will be going is as shown below</p>

![image](https://user-images.githubusercontent.com/88897287/139188519-064d4dea-c7ab-41a3-85f5-2a2afbba8a2f.png)

<h2>Execution</h2>

<p align="center">
  <img src="https://user-images.githubusercontent.com/88897287/138466372-fa1ada57-8df2-42be-ac03-f71c0daa3b80.png">
</p>


<h2>Outcome</h2>


<p>
  The image below is an illustration of third component in PCA illustrated in SIMCA, which shows significant patterns during process change/transitions. I display third component (t[3]) becuase t[1] and t[2] plot were not really that insightful
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/88897287/138473642-2c9cf64b-5c64-48c8-bcfa-ad7ee5860ceb.png">
</p>

<p>
  This t[3] also tells us at what time 'process drift' occurs, which after clarification with related personned, the process drift occurs during plant turnaround. Thus in the future, these data will be omitted
</p>


<p align="center">
  <img src="https://user-images.githubusercontent.com/88897287/138474662-78fe2b69-926b-41c5-989d-d30ca1bafe71.png">
</p>


<h2>References</h2>

<p>
  Schmalen, P. (2020, August 16). Understand your data with principal component analysis (PCA) and discover underlying patterns. Medium. Retrieved October 22, 2021, from https://towardsdatascience.com/understand-your-data-with-principle-component-analysis-pca-and-discover-underlying-patterns-d6cadb020939. 
  
Massaron, L., &amp; Mueller, J. P. (2019). In Machine learning for dummies (pp. 230–232).
  
  </p>
