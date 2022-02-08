from classifier import *
from models import *

class Look_Ahead_model(nn.Module):
    def __init__(self):
        super(Look_Ahead_model, self).__init__()
        self.actor = Actor()
        self.sensor = Sensor()
        self.aggregator = Aggregator()
        self.lookahead = LookAhead()
        self.classifier = Classifier()

    def forward(self, a_before, p_before, x):
        m_now = self.actor(a_before,p_before)
        p_now = p_before + m_now #Change
        # Get new view x
        a_now = self.sensor(m_now, x)
        a_now = self.aggregator(a_now)
        a_after = self.lookahead(a_now, m_now, p_now)
        #Delay
        y_predicted = self.classifier(a_now)
        return a_now, a_after, y_predicted

def loss(a_now, a_after):
    #Cosine distance = 1- cosine similarity
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    loss = cos(a_now, a_after)
    return 1-loss