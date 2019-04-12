import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools

import plotly.offline as pltoff
pltoff.init_notebook_mode(connected=True)

from IPython.core.display import display
import IPython

import numpy as np
from sklearn.manifold import TSNE

# https://stackoverflow.com/questions/47230817/plotly-notebook-mode-with-google-colaboratory
# https://stackoverflow.com/questions/51119951/interactive-graph-in-colab

def configure_plotly_browser_state():
    display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              plotly: 'https://cdn.plot.ly/plotly-1.5.1.min.js?noext',
            },
          });
        </script>
        '''))

def _word_viz(fitted, labels):
  configure_plotly_browser_state()

  plot = [go.Scatter3d(x = fitted[:, 0],
                      y = fitted[:, 1],
                      z = fitted[:, 2],
                      mode = 'markers+text',
                      text = labels,
                      textposition='bottom center',
                      hoverinfo = 'text',
                      marker=dict(size=2.5,opacity=0.8)
                      )]

  layout = go.Layout(title='Youtube Titles')
  fig = go.Figure(data=plot, layout=layout)
  pltoff.iplot(fig)


def word_viz(model, top_n = 1000):
    words = list(model.vocab.word2index.keys())[:top_n]
    vecs = np.array([model[w].detach().numpy() for w in words])

    tsne_model = TSNE(perplexity=45, n_components=3, init='pca', n_iter=5000, random_state=16)
    tsne_vecs = tsne_model.fit_transform(vecs.reshape((top_n, 200)))
    
    to_display = top_n // 10
    _word_viz(tsne_vecs[:to_display], words[:to_display])
    
    return tsne_vecs
