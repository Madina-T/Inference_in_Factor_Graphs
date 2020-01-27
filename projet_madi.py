import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
import pydotplus as dot
from IPython.display import SVG
import numpy as np
import random
import copy
import sys

# Inférences dans un Factor Graph

class FactorGraph:
    
    def __init__(self):
        self.variables = [] # liste des noms des variables
        self.factors = [] # liste des noms des facteurs
        self.variables_names = {} # dictionnaire { nom_variable : variable }
        self.factors_names = {} # dictionnaire { nom_facteur : facteur }
        self.edges = [] # liste des arêtes
        self.variables_neighbours = {} # dictionnaire { nom_variable : liste des noms
                                       # de ses voisins }
        self.factors_neighbours = {} # dictionnaire { nom_facteur : liste des noms de
                                     # ses voisins }
        self.bn = None # réseau bayésien associé au Factor Graph

    def addVariable(self, v):
        """
        Fonction permettant d'ajouter une variable au Factor Graph
        Entrée :
            v : variable de type gum.DiscreteVariable
        """
        self.variables.append(v.name())
        self.variables_names[v.name()] = v
        
    def addFactor(self, f):
        """
        Fonction permettant d'ajouter un facteur au Factor Graph
        Entrées :
            f : facteur de type gum.Potential
        """
        vars_names = [var.name() for var in f.variablesSequence()]
        f_name = "p" + str(vars_names[0])
        if len(vars_names) > 1:
            f_name += "g" + "".join(vars_names[1:])
        self.factors.append(f_name)
        self.factors_names[f_name] = f

    def addEvidence(self, evidence):
        """
        Fonction permettant d'ajouter des observations sur une ou plusieurs 
        variables dans le Factor Graph ( une observation sur une variable prend
        la forme d'un facteur supplémentaire dans le factor graph )
        Entrée :
            evidence : dictionnaire d'observations sur une ou plusieurs variables
            { variable X : valeur observée pour X }
        """
        for v_name, v_value in evidence.items():
            new_factor = gum.Potential().add(self.variables_names[v_name])
            new_factor.fillWith(0)
            new_factor[v_value] = 1
            new_factor_name = "p"+v_name+"_new"
            self.factors.append(new_factor_name)
            self.factors_names[new_factor_name] = new_factor
            self.edges.append((new_factor_name,v_name))
            self.variables_neighbours[v_name].append(new_factor_name)
            self.factors_neighbours[new_factor_name] = [v_name]
        
    def build(self,bn):
        """
        Fonction permettant de construire un Factor Graph à partir d'un réseau bayésien
        Entrée :
            bn : réseau bayésien de type gum.BayesNet
        """
        self.bn = bn
        for i in bn.nodes():
            self.addVariable(bn.variable(i))
            f = bn.cpt(i)
            self.addFactor(f)
            vars_names = [var.name() for var in f.variablesSequence()]
            f_name = self.factors[-1]
            for v in vars_names:
                self.edges.append((f_name,v))
                if v not in self.variables_neighbours:
                    self.variables_neighbours[v] = [f_name]
                else:
                    self.variables_neighbours[v].append(f_name)
            self.factors_neighbours[f_name] = f.var_names
    
    def copy(self):
        """
        Fonction permettant de créer une copie du Factor Graph
        Sortie:
            fg_copy : copie du Factor Graph
        """
        fg_copy = FactorGraph()
        fg_copy.build(self.bn)
        return fg_copy
    
    def show(self):
        """
        Fonction permettant de visualiser le Factor Graph construit
        """
        variables = ""
        for v in self.variables:
            variables += v + ";"

        factors = ""
        for f_name in self.factors:
            f = self.factors_names[f_name]
            vars_names = [v.name() for v in f.variablesSequence()]
            f_name = "p" + str(vars_names[0])
            if len(vars_names) > 1:
                f_name += "g" + "".join(vars_names[1:])
            factors += f_name + ";\n"

        links = ""
        for f, v in self.edges:
            links += f + "--" + v + ";"
            links += "\n"

        dot_code = """graph FG {\nlayout=neato;\n\n//les variables\nnode [shape=rectangle,margin=0.04,
          width=0,height=0, style=filled,color="coral"];\n""" + variables + """\n\n//les facteurs\nnode [shape=point,width=0.1,height=0.1, style=filled,color="burlywood"];\n""" + factors + """\n//les liens variable--facteurs\nedge [len=0.7];\n\n""" + links + """}"""

        fg = dot.graph_from_dot_data(dot_code)
        
        return SVG(fg.create_svg())

# Inférence Sum-Product

class TreeSumProductInference:

    def __init__(self, fg):
        self.fg = fg.copy() # on travaille avec une copie du Factor Graph pour ne pas
                            # l'altérer
        self.messages = {} # dictionnaire dans lequel on stocke l'ensemble des messages sous
                           # la forme { (émetteur, récepteur) : message }

    def makeInference(self):
        """
        Fonction effectuant le calcul de tous les messages ( algorithme Sum-Product )
        """
        def send_variable_message(x_m):
            # x_m envoie un message à chacun de ses voisins se trouvant après lui dans
            # l'ordre défini ( self.order )
            x_m_neighbours = [f for f in self.fg.variables_neighbours[x_m] if f not in order[:order.index(x_m)]]
            for f_s in x_m_neighbours:
                # print(f_s)
                # si x_m est une feuille, elle envoie 1 à f_s
                if x_m in self.leaves:
                    message = gum.Potential()
                else:
                    neighbours = [f for f in self.fg.variables_neighbours[x_m] if f != f_s] # voisins de x_m différents de f_s
                    # on fait le produit de tous les messages venant de chaque voisin de
                    # x_m différent de f_s
                    message = gum.Potential()
                    for f_n in neighbours: # f_n : facteur voisin
                        message *= self.messages[(f_n,x_m)]
                self.messages[(x_m,f_s)] = message
                # gnb.showPotential(self.messages[(x_m,f_s)])

        def send_factor_message(f_s):
            # f_s envoie un message à chacun de ses voisins après lui dans l'ordre défini
            # ( self.order )
            f_s_neighbours = [v for v in self.fg.factors_neighbours[f_s] if v not in order[:order.index(f_s)]]
            for x in f_s_neighbours:
                # print(x)
                # si f_s est une feuille, il envoie son contenu ( sa table de probabilité 
                # conditionnelle ) à x
                if f_s in self.leaves:
                    message = gum.Potential(self.fg.factors_names[f_s])
                else:
                    # sinon
                    neighbours = [v for v in self.fg.factors_neighbours[f_s] if v != x] # voisins de f_s différents de x
                    # on instancie le message comme étant la table de probabilité
                    # conditionnelle contenue dans f_s
                    message = gum.Potential(self.fg.factors_names[f_s])
                    if neighbours:
                        # on le multiplie par le produit de tous les messages venant 
                        # de chaque voisin de f_s différent de x 
                        for x_n in neighbours: # x_n : variable voisine
                            message *= self.messages[(x_n,f_s)]
                # on somme ce produit sur toute les variables de f_s différentes
                # de x
                message = message.margSumIn([x])
                self.messages[(f_s,x)] = message
                # gnb.showPotential(self.messages[(f_s,x)])

        # On détermine les feuilles du Factor Tree ( que l'on doit distinguer des autres
        # sommets pour l'envoi des messages )
        self.leaves = [] # feuilles de l'arbre
        for vertex, neighbours in self.fg.factors_neighbours.items():
            if len(neighbours) == 1:
                self.leaves.append(vertex)
        for vertex, neighbours in self.fg.variables_neighbours.items():
            if len(neighbours) == 1:
                self.leaves.append(vertex)

        # On définit l'ordre d'envoi des messages ( sommets de degré croissant en
        # décrémentant le degré d'un sommet dont un voisin apparaît dans l'ordre )
        # De cette manière, les messages seront envoyés des feuilles de l'arbre
        # vers la racine
        vertices = self.fg.variables + self.fg.factors
        nb_neighbours = [len(self.fg.variables_neighbours[variable]) for variable in self.fg.variables]
        nb_neighbours += [len(self.fg.factors_neighbours[factor]) for factor in self.fg.factors]
        self.order = []
        self.root = None
        cpt = 0 # compteur : si on ne trouve plus de variable avec un unique voisin, on 
                # arrête
        while len(self.order) < len(vertices):
            new_cpt = cpt
            cpt = 0
            if len(self.order) == len(vertices)-1:
                for i in range(len(vertices)):
                    if vertices[i] not in self.order:
                        self.order.append(vertices[i])
            for i in range(len(vertices)):
                if len(self.order) == len(vertices)-2:
                    for i in range(len(vertices)):
                        if vertices[i] in self.fg.factors and vertices[i] not in self.order:
                            self.order.append(vertices[i])
                if nb_neighbours[i] == 1 and vertices[i] not in self.order:
                    self.order.append(vertices[i]);
                    nb_neighbours[i] = 0
                    for j in range(len(vertices)):
                        if vertices[i] in self.fg.variables and vertices[j] in self.fg.variables_neighbours[vertices[i]] or vertices[i] in self.fg.factors and vertices[j] in self.fg.factors_neighbours[vertices[i]]:
                            nb_neighbours[j] -= 1
                else:
                    cpt += 1
            if 1 not in nb_neighbours and new_cpt == cpt:
                break

        not_in_order = [vertex for vertex in vertices if vertex not in self.order]

        # S'il reste des sommets n'ayant pas été ajouté dans self.order cela signifie que
        # fg n'est pas un arbre
        if not_in_order:
            print("Attention, cet algorithme ne peut être appliqué qu'à des arbres",file=sys.stderr)
            return

        self.root = self.order[-1]
        
        # print("\nOrdre d'envoi :",self.order)
        #print("\nRacine:",self.root)

        # On envoie les messages des feuilles vers la racine
        #print("\nFeuilles -> Racine\n")
        order = self.order[:-1]
        #print(order)
        for vertex in order:
            if vertex in self.fg.variables:
                # print(vertex,"envoie son message à")
                send_variable_message(vertex)
            elif vertex in self.fg.factors:
                # print(vertex,"envoie son message à")
                send_factor_message(vertex)

        # On envoie les messages de la racine vers les feuilles
        #print("\nRacine -> Feuilles\n")
        order = list(reversed(self.order))
        #print(order)
        for vertex in order:
            if vertex in self.fg.variables:
                # print(vertex,"envoie son message à")
                send_variable_message(vertex)
            elif vertex in self.fg.factors:
                # print(vertex,"envoie son message à")
                send_factor_message(vertex)
        
    def posterior(self, v_name):
        """
        Fonction permettant de déterminer la distribution de la variable v_name 
        Entrée:
            v_name : nom de la variable dont on souhaite connaître la distribution
        Sortie:
            posterior : distribution de la variable v_name
        """
        # On fait le produit de tous les messages reçus par v_name ( messages calculés dans
        # makeInference()) puis on normalise le produit obtenu
        posterior = gum.Potential()
        for (sender,receiver), message in self.messages.items():
            if receiver == v_name:
                posterior *= message
        return posterior.normalize()

    def addEvidence(self, evidence):
        """
        Fonction permettant de rajouter des observations sur une ou plusieurs 
        variables dans le Factor Graph fg
        Entrée :
            evidence : dictionnaire d'observations sur une ou plusieurs variables
            { variable X : valeur observée pour X }
        """
        self.fg.addEvidence(evidence)
        
# Inférence Max-Product

class TreeMaxProductInference:
    
    def __init__(self, fg):
        self.fg = fg.copy() # on travaille avec une copie du Factor Graph pour ne pas
                            # l'altérer
        self.messages = {} # dictionnaire dans lequel on stocke l'ensemble des messages sous
                           # la forme { (émetteur, récepteur) : message }
        self.phi = {} # dictionnaire stockant, pour chaque modalité de chaque variable, 
                      # la valeur la plus probable des variables voisines ( avec lesquelles 
                      # elle a un facteur commun ) pour la probabilité jointe ( produit des
                      # messages qu'elle a reçu )
                      # Le dictionnaire prend la forme suivante :
                      # { ( variable X, valeur de X ) : { variable voisine X_n : valeur 
                      # la plus probable de X_n pour la probabilité jointe quand X = valeur
                      # de X } }

    def makeInference(self):
        """
        Fonction effectuant le calcul de tous les messages ( algorithme Max-Product )
        """
        def send_variable_message(x_m):
            # x_m envoie un message à chacun de ses voisins se trouvant après lui dans
            # l'ordre défini
            x_m_neighbours = [f for f in self.fg.variables_neighbours[x_m] if f not in order[:order.index(x_m)]]
            for f_s in x_m_neighbours:
                # print(f_s)
                # si x_m est une feuille, elle envoie 1 à f_s
                if x_m in self.leaves:
                    message = gum.Potential()
                else:
                    neighbours = [f for f in self.fg.variables_neighbours[x_m] if f != f_s] # voisins de x_m différents de f_s
                    # on fait le produit de tous les messages venant de chaque voisin de 
                    # x_m différent de f_s
                    message = gum.Potential()
                    for f_n in neighbours: # f_n : facteur voisin
                        message *= gum.Potential(self.messages[(f_n,x_m)])
                self.messages[(x_m,f_s)] = message
                # gnb.showPotential(self.messages[(x_m,f_s)])

        def send_factor_message(f_s):
            # f_s envoie un message à chacun de ses voisins après lui dans l'ordre défini
            f_s_neighbours = [v for v in self.fg.factors_neighbours[f_s] if v not in order[:order.index(f_s)]]
            for x in f_s_neighbours:
                # print(x)
                # si f_s est une feuille, il envoie son contenu ( table de probabilité
                # conditionnelle ) à x
                if f_s in self.leaves:
                    message = gum.Potential(self.fg.factors_names[f_s])
                else:
                    neighbours = [v for v in self.fg.factors_neighbours[f_s] if v != x] # voisins de f_s différents de x
                    # on instancie le message comme étant la table de probabilité
                    # conditionnelle contenue dans f_s
                    message = gum.Potential(self.fg.factors_names[f_s])
                    if neighbours:
                        # on le multiplie par le produit de tous les messages venant de
                        # chaque voisin de f_s différent de x 
                        for x_n in neighbours: # x_n : variable voisine
                            message *= gum.Potential(self.messages[(x_n,f_s)])
                # on stocke dans phi la configuration des variables différentes de x 
                # ( parmi ses variables voisines i.e. séparées d'elle par un facteur ) 
                # ayant donné le maximum pour chaque modalité de x
                if len(message.variablesSequence()) > 1:
                    I = gum.Instantiation()
                    for v in message.variablesSequence():
                        if v.name() == x:
                            I.add(v)
                            for x_value in list(map(int,v.domain()[1:-1].split(','))):
                                I[x] = x_value
                                phi = message.extract(I).argmax() # configuration des 
                                                                  # variables différentes de
                                                                  # x ayant donné le maximum
                                                                  # pour x = x_value
                                if (x,x_value) in self.phi:
                                    for name, value in phi[0].items():
                                        self.phi[(x,x_value)][name] = value
                                else:
                                    self.phi[(x,x_value)] = phi[0]
                # on effectue un max sur toutes les variables de f_s différentes de x
                message = message.margMaxIn([x])
                self.messages[(f_s,x)] = message
                # gnb.showPotential(self.messages[(f_s,x)])

        # On détermine les feuilles du Factor Tree ( que l'on doit distinguer des autres
        # sommets pour l'envoi des messages )
        self.leaves = [] # feuilles de l'arbre
        for vertex, neighbours in self.fg.factors_neighbours.items():
            if len(neighbours) == 1:
                self.leaves.append(vertex)
        for vertex, neighbours in self.fg.variables_neighbours.items():
            if len(neighbours) == 1:
                self.leaves.append(vertex)

        # On définit l'ordre d'envoi des messages ( sommets de degré croissant en
        # décrémentant le degré d'un sommet dont un voisin apparaît dans l'ordre )
        # De cette manière, les messages seront envoyés des feuilles de l'arbre
        # vers la racine
        vertices = self.fg.variables + self.fg.factors
        nb_neighbours = [len(self.fg.variables_neighbours[variable]) for variable in self.fg.variables]
        nb_neighbours += [len(self.fg.factors_neighbours[factor]) for factor in self.fg.factors]
        self.order = []
        self.root = None
        cpt = 0 # compteur : si on ne trouve plus de variable avec un unique voisin, on 
                # arrête
        while len(self.order) < len(vertices):
            new_cpt = cpt
            cpt = 0
            if len(self.order) == len(vertices)-1:
                for i in range(len(vertices)):
                    if vertices[i] not in self.order:
                        self.order.append(vertices[i])
            for i in range(len(vertices)):
                if len(self.order) == len(vertices)-2:
                    for i in range(len(vertices)):
                        if vertices[i] in self.fg.factors and vertices[i] not in self.order:
                            self.order.append(vertices[i])
                if nb_neighbours[i] == 1 and vertices[i] not in self.order:
                    self.order.append(vertices[i]);
                    nb_neighbours[i] = 0
                    for j in range(len(vertices)):
                        if vertices[i] in self.fg.variables and vertices[j] in self.fg.variables_neighbours[vertices[i]] or vertices[i] in self.fg.factors and vertices[j] in self.fg.factors_neighbours[vertices[i]]:
                            nb_neighbours[j] -= 1
                else:
                    cpt += 1
            if 1 not in nb_neighbours and new_cpt == cpt:
                break

        not_in_order = [vertex for vertex in vertices if vertex not in self.order]

        # S'il reste des sommets n'ayant pas été ajouté dans self.order cela signifie que
        # fg n'est pas un arbre
        if not_in_order:
            print("Attention, cet algorithme ne peut être appliqué qu'à des arbres",file=sys.stderr)
            return

        self.root = self.order[-1]
        
        # print("\nOrdre d'envoi :",self.order)
        # print("\nRacine:",self.root)

        # On envoie les messages des feuilles vers la racine
        order = self.order[:-1]
        #print(order)
        for vertex in order:
            if vertex in self.fg.variables:
                # print(vertex,"envoie son message à")
                send_variable_message(vertex)
            elif vertex in self.fg.factors:
                # print(vertex,"envoie son message à")
                send_factor_message(vertex)

    def argmax(self):
        """
        Fonction permettant de déterminer la valeur la plus probable de chaque variable
        du Factor Graph pour la probabilité jointe
        Sortie :
            argmax : dictionnaire des valeurs des variables les plus probables pour la 
            probabilité jointe
        """
        argmax = {}
        if self.root: # si makeInference() a fonctionné correctement, une racine a été
                      # définie
            # on commence par déterminer la valeur la plus probable de la racine en calculant
            # l'argmax du produit des messages qu'elle a reçu
            m_r = gum.Potential()
            compute_phi = gum.Potential()
            for (sender,receiver), message in self.messages.items():
                if receiver == self.root:
                    m_r *= message
            argmax = m_r.argmax()[0]

            # print(self.phi)

            # Back-tracking de la racine aux feuilles pour récupérer la valeur la plus 
            # probable de chaque variable
            order = [vertex for vertex in list(reversed(self.order)) if vertex in self.fg.variables and vertex != self.root]
            parent = self.root
            for vertex in order:
                for tpl in self.phi:
                    if tpl[0] == parent and parent in argmax and tpl[1] == argmax[parent]:
                        for v_name, v_value in self.phi[tpl].items():
                            argmax[v_name] = v_value
                parent = vertex

        return argmax

    def addEvidence(self, evidence):
        """
        Fonction permettant de rajouter des observations sur une ou plusieurs 
        variables dans le Factor Graph fg
        Entrée :
            evidence : dictionnaire d'observations sur une ou plusieurs variables
            { variable X : valeur observée pour X }
        """
        self.fg.addEvidence(evidence)

# Inférence Max-Sum

class TreeMaxSumInference:
    
    def __init__(self, fg):
        self.fg = fg.copy() # on travaille avec une copie du Factor Graph pour ne pas
                            # l'altérer
        self.messages = {} # dictionnaire dans lequel on stocke l'ensemble des messages sous
                           # la forme { (émetteur, récepteur) : message }
        self.phi = {} # dictionnaire stockant, pour chaque modalité de chaque variable, 
                      # la valeur la plus probable des variables voisines ( avec lesquelles 
                      # elle a un facteur commun ) pour la probabilité jointe ( produit des
                      # messages qu'elle a reçu )
                      # Le dictionnaire prend la forme suivante :
                      # { ( variable X, valeur de X ) : { variable voisine X_n : valeur 
                      # la plus probable de X_n pour la probabilité jointe quand X = valeur
                      # de X } }

    def makeInference(self):
        """
        Fonction effectuant le calcul de tous les messages ( algorithme Max-Sum )
        """
        def send_variable_message(x_m):
            # x_m envoie un message à chacun de ses voisins se trouvant après lui dans
            # l'ordre défini
            x_m_neighbours = [f for f in self.fg.variables_neighbours[x_m] if f not in order[:order.index(x_m)]]
            for f_s in x_m_neighbours:
                # print(f_s)
                # si x_m est une feuille, elle envoie 0 à f_s
                if x_m in self.leaves:
                    message = gum.Potential().fillWith(0)
                else:
                    neighbours = [f for f in self.fg.variables_neighbours[x_m] if f != f_s] # voisins de x_m différents de f_s
                    # on fait la sommme de tous les messages ( log probas ) venant de chaque
                    # voisin de x_m différent de f_s
                    message = gum.Potential().fillWith(0)
                    for f_n in neighbours: # f_n : facteur voisin
                        message += self.messages[(f_n,x_m)]
                self.messages[(x_m,f_s)] = message
                # gnb.showPotential(self.messages[(x_m,f_s)])

        def send_factor_message(f_s):
            # f_s envoie un message à chacun de ses voisins après lui dans l'ordre défini
            f_s_neighbours = [v for v in self.fg.factors_neighbours[f_s] if v not in order[:order.index(f_s)]]
            for x in f_s_neighbours:
                # print(x)
                # si f_s est une feuille, il envoie son contenu ( le log de sa table de
                # probabilité conditionnelle ) à x
                if f_s in self.leaves:
                    message = gum.Potential(self.fg.factors_names[f_s])
                    for i in message.loopIn():
                        if message.get(i) != 0:
                            message.set(i,np.log(message.get(i))) # on passe au log ( ln )
                        else:
                            message.set(i,-1000000000)  # étant donné que ln(x) tend vers 
                                                        # - l'infini lorsque x tend vers 0,
                                                        # on remplace ici les 0 par 
                                                        # -1000000000
                                                        # Remarque : une valeur encore plus
                                                        # petite aurait pu être choisie
                else:
                    neighbours = [v for v in self.fg.factors_neighbours[f_s] if v != x] # voisins de f_s différents de x
                    # on instancie le message comme étant le log de la table de probabilité 
                    # conditionnelle contenue dans f_s
                    message = gum.Potential(self.fg.factors_names[f_s])
                    for i in message.loopIn():
                        if message.get(i) != 0:
                            message.set(i,np.log(message.get(i))) # on passe au log (ln)
                    if neighbours:
                        # on fait la sommme de tous les messages venant de chaque voisin
                        # de f_s différent de x ( log probas )
                        for x_n in neighbours: # x_n : variable voisine
                            message += gum.Potential(self.messages[(x_n,f_s)])
                # on passe à des valeurs positives pour que Potential.argmax() retourne le
                # bon dictionnaire
                message_abs = gum.Potential(message)
                min_cpt = message_abs.min()
                if min_cpt <= 0:
                    for i in message_abs.loopIn():
                        message_abs.set(i,message_abs.get(i) - min_cpt)
                message = message_abs
                # on stocke dans phi la configuration des variables différentes de x 
                # ( parmi ses variables voisines i.e. séparées d'elle par un facteur ) 
                # ayant donné le maximum pour chaque modalité de x
                if len(message.variablesSequence()) > 1:
                    I = gum.Instantiation()
                    for v in message.variablesSequence():
                        if v.name() == x:
                            I.add(v)
                            for x_value in list(map(int,v.domain()[1:-1].split(','))):
                                I[x] = x_value
                                phi = message.extract(I).argmax() # configuration des 
                                                                  # variables différentes de
                                                                  # x ayant donné le maximum
                                                                  # pour x = x_value
                                if (x,x_value) in self.phi:
                                    for name, value in phi[0].items():
                                        self.phi[(x,x_value)][name] = value
                                else:
                                    self.phi[(x,x_value)] = phi[0]
                # on fait un max sur toutes les variables de f_s différentes de x
                message = message.margMaxIn([x])
                self.messages[(f_s,x)] = message
                # gnb.showPotential(self.messages[(f_s,x)])

        # On détermine les feuilles du Factor Tree ( que l'on doit distinguer des autres
        # sommets pour l'envoi des messages )
        self.leaves = [] # feuilles de l'arbre
        for vertex, neighbours in self.fg.factors_neighbours.items():
            if len(neighbours) == 1:
                self.leaves.append(vertex)
        for vertex, neighbours in self.fg.variables_neighbours.items():
            if len(neighbours) == 1:
                self.leaves.append(vertex)

        # On définit l'ordre d'envoi des messages ( sommets de degré croissant en
        # décrémentant le degré d'un sommet dont un voisin apparaît dans l'ordre )
        # De cette manière, les messages seront envoyés des feuilles de l'arbre
        # vers la racine
        vertices = self.fg.variables + self.fg.factors
        nb_neighbours = [len(self.fg.variables_neighbours[variable]) for variable in self.fg.variables]
        nb_neighbours += [len(self.fg.factors_neighbours[factor]) for factor in self.fg.factors]
        self.order = []
        self.root = None
        cpt = 0 # compteur : si on ne trouve plus de variable avec un unique voisin, on 
                # arrête
        while len(self.order) < len(vertices):
            new_cpt = cpt
            cpt = 0
            if len(self.order) == len(vertices)-1:
                for i in range(len(vertices)):
                    if vertices[i] not in self.order:
                        self.order.append(vertices[i])
            for i in range(len(vertices)):
                if len(self.order) == len(vertices)-2:
                    for i in range(len(vertices)):
                        if vertices[i] in self.fg.factors and vertices[i] not in self.order:
                            self.order.append(vertices[i])
                if nb_neighbours[i] == 1 and vertices[i] not in self.order:
                    self.order.append(vertices[i]);
                    nb_neighbours[i] = 0
                    for j in range(len(vertices)):
                        if vertices[i] in self.fg.variables and vertices[j] in self.fg.variables_neighbours[vertices[i]] or vertices[i] in self.fg.factors and vertices[j] in self.fg.factors_neighbours[vertices[i]]:
                            nb_neighbours[j] -= 1
                else:
                    cpt += 1
            if 1 not in nb_neighbours and new_cpt == cpt:
                break

        not_in_order = [vertex for vertex in vertices if vertex not in self.order]

        # S'il reste des sommets n'ayant pas été ajouté dans self.order cela signifie que
        # fg n'est pas un arbre
        if not_in_order:
            print("Attention, cet algorithme ne peut être appliqué qu'à des arbres",file=sys.stderr)
            return

        self.root = self.order[-1]
        
        # print("\nOrdre d'envoi :",self.order)
        #print("\nRacine :",self.root)

        # On envoie des messages des feuilles vers la racine
        order = self.order[:-1]
        #print(order)
        for vertex in order:
            if vertex in self.fg.variables:
                # print(vertex,"envoie son message à")
                send_variable_message(vertex)
            elif vertex in self.fg.factors:
                # print(vertex,"envoie son message à")
                send_factor_message(vertex)

    def argmax(self):
        """
        Fonction permettant de déterminer la valeur la plus probable de chaque variable
        du Factor Graph pour la probabilité jointe
        Sortie :
            argmax : dictionnaire des valeurs des variables les plus probables pour la 
            probabilité jointe
        """
        argmax = {}
        if self.root: # si makeInference() a fonctionné correctement, une racine a été
                      # définie
            # on commence par déterminer la valeur la plus probable de la racine en calculant
            # l'argmax de la somme des messages qu'elle a reçu
            m_r = gum.Potential().fillWith(0)
            compute_phi = gum.Potential().fillWith(0)
            for (sender,receiver), message in self.messages.items():
                if receiver == self.root:
                    m_r += message
            argmax = m_r.argmax()[0]

            #print(self.phi)

            # Back-tracking de la racine aux feuilles pour récupérer la valeur la plus 
            # probable de chaque variable
            order = [vertex for vertex in list(reversed(self.order)) if vertex in self.fg.variables and vertex != self.root]
            parent = self.root
            for vertex in order:
                for tpl in self.phi:
                    if tpl[0] == parent and parent in argmax and tpl[1] == argmax[parent]:
                        for v_name, v_value in self.phi[tpl].items():
                            argmax[v_name] = v_value
                parent = vertex

        return argmax

    def addEvidence(self, evidence):
        """
        Fonction permettant de rajouter des observations sur une ou plusieurs 
        variables dans le Factor Graph fg
        Entrée :
            evidence : dictionnaire d'observations sur une ou plusieurs variables
            { variable X : valeur observée pour X }
        """
        self.fg.addEvidence(evidence)

# Inférence Loopy Belief Propagation Sum-Product

class LBPSumProductInference:
    
    def __init__(self, fg):
        self.fg = fg.copy() # on travaille avec une copie du Factor Graph pour ne pas
                            # l'altérer
        self.messages = {} # dictionnaire dans lequel on stocke l'ensemble des messages sous
                           # la forme { (émetteur, récepteur) : message }

    def makeInference(self):
        """
        Fonction effectuant le calcul de tous les messages ( algorithme Loopy Belief 
        Propagation Sum-Product )
        """
        def send_variable_message(x_m):
            new_message = False
            # x_m envoie un message à chacun de ses voisins
            x_m_neighbours = self.fg.variables_neighbours[x_m]
            for f_s in x_m_neighbours:
                # print(f_s)
                neighbours = [f for f in self.fg.variables_neighbours[x_m] if f != f_s] # voisins de x_m différents de f_s
                # on fait le produit de tous les messages venant de chaque voisin de x_m 
                # différent de f_s
                message = gum.Potential()
                for f_n in neighbours: # f_n : facteur voisin
                    message *= gum.Potential(self.messages[(f_n,x_m)])
                # print("Avant")
                # gnb.showPotential(self.messages[(x_m,f_s)])
                if self.messages[(x_m,f_s)] != message:
                    new_message = True
                    self.messages[(x_m,f_s)] = message
                    # print("Après")
                    # gnb.showPotential(self.messages[(x_m,f_s)])
            return new_message

        def send_factor_message(f_s):
            new_message = False
            # f_s envoie un message à chacun de ses voisins
            f_s_neighbours = self.fg.factors_neighbours[f_s]
            for x in f_s_neighbours:
                # print(x)
                if f_s in self.leaves:
                    message = gum.Potential(self.fg.factors_names[f_s])
                else:
                    neighbours = [v for v in self.fg.factors_neighbours[f_s] if v != x] # voisins de f_s différents de x
                    # on instancie le message comme étant la table de probabilité 
                    # conditionnelle contenue dans f_s
                    message = gum.Potential(self.fg.factors_names[f_s])
                    if neighbours:
                        # on le multiplie par le produit de tous les messages venant de
                        # chaque voisin de f_s différent de x 
                        for x_n in neighbours: # x_n : variable voisine
                            message *= gum.Potential(self.messages[(x_n,f_s)])
                # on somme ce produit sur toute les variables de f_s différentes de x
                message = message.margSumIn([x])
                # print("Avant")
                # gnb.showPotential(self.messages[(f_s,x)])
                if self.messages[(f_s,x)] != message.normalize():
                    new_message = True
                    self.messages[(f_s,x)] = message
                    # print("Après")
                    # gnb.showPotential(self.messages[(f_s,x)])
            return new_message

        # On détermine les feuilles du Factor Graph s'il en a
        self.leaves = [] # feuilles
        for vertex, neighbours in self.fg.factors_neighbours.items():
            if len(neighbours) == 1:
                self.leaves.append(vertex)
        for vertex, neighbours in self.fg.variables_neighbours.items():
            if len(neighbours) == 1:
                self.leaves.append(vertex)

        # On définit le début de l'ordre d'envoi des messages ( sommets de degré croissant
        # en décrémentant le degré d'un sommet dont un voisin apparaît dans l'ordre )
        # Si le Factor Graph est un arbre, tous les sommets seront ajouter à self.order
        # durant cette étape
        vertices = self.fg.variables + self.fg.factors
        nb_neighbours = [len(self.fg.variables_neighbours[variable]) for variable in self.fg.variables] # nombre de voisins dont on n'a pas encore recu de message
        nb_neighbours += [len(self.fg.factors_neighbours[factor]) for factor in self.fg.factors]
        self.order = []
        self.root = None
        cpt = 0 # compteur : si on ne trouve plus de variable avec un unique voisin, on 
                # arrête
        while len(self.order) < len(vertices):
            new_cpt = cpt
            cpt = 0
            if len(self.order) == len(vertices)-1:
                for i in range(len(vertices)):
                    if vertices[i] not in self.order:
                        self.order.append(vertices[i])
            for i in range(len(vertices)):
                if len(self.order) == len(vertices)-2:
                    for i in range(len(vertices)):
                        if vertices[i] in self.fg.factors and vertices[i] not in self.order:
                            self.order.append(vertices[i])
                if nb_neighbours[i] == 1 and vertices[i] not in self.order:
                    self.order.append(vertices[i]);
                    nb_neighbours[i] = 0
                    for j in range(len(vertices)):
                        if vertices[i] in self.fg.variables and vertices[j] in self.fg.variables_neighbours[vertices[i]] or vertices[i] in self.fg.factors and vertices[j] in self.fg.factors_neighbours[vertices[i]]:
                            nb_neighbours[j] -= 1
                else:
                    cpt += 1
            if new_cpt == cpt and len(self.order) < len(vertices)-1:
                break
        not_in_order = [vertex for vertex in vertices if vertex not in self.order]

        # S'il reste des sommets n'ayant pas été ajouté dans self.order cela signifie que
        # fg n'est pas un arbre
        if not_in_order:
            # on choisit alors une racine au hasard parmi les variables restantes
            self.root = random.choice([vertex for vertex in not_in_order if vertex in self.fg.variables])
            # on détermine un chemin de chacun des sommets restants à cette racine
            self.path_to_root = {}
            self.path_to_root[self.root] = None # parcours sous forme d'un dictionnaire { noeud n : père de n }
            Q = [self.root] # file FIFO ( on enfile un sommet lorsqu'il est découvert et on le défile lorsqu'on a terminé de le traiter )
            while Q:
                u = Q.pop(0)
                if u in self.fg.variables:
                    neighbours = self.fg.variables_neighbours[u]
                elif u in self.fg.factors:
                    neighbours = self.fg.factors_neighbours[u]
                for v in neighbours:
                    if v not in self.path_to_root:
                        self.path_to_root[v]=u
                        Q.append(v)

            # on détermine les feuilles du sous graphe ne contenant que les sommets n'ayant
            # pas encore été ajoutés dans self.order 
            leaves_not_in_order = []
            for vertex in not_in_order:
                if vertex in self.fg.variables:
                    for v,parent in self.path_to_root.items():
                        if parent == vertex and v in self.order:
                            if vertex not in leaves_not_in_order:
                                leaves_not_in_order.append(vertex)

            # on ajoute ensuite les sommets restants des feuilles du sous graphe jusqu'à
            # la racine
            while len(self.order) < len(vertices):
                if leaves_not_in_order:
                    vertex = leaves_not_in_order.pop()
                    while vertex != None and vertex not in self.order and vertex != self.root:
                        self.order.append(vertex)
                        vertex = self.path_to_root[vertex]
                else:
                    if len(self.order) == len(vertices) - 1:
                        self.order.append(self.root)
                    for vertex in vertices:
                        if vertex not in self.order and vertex != self.root:
                            self.order.append(vertex)

        self.root = self.order[-1]

        #print("\nOrdre d'envoi :",self.order)
        #print("\nRacine:",self.root)

        # Initialisation des messages ( chaque sommet doit être en capacité d'envoyer un 
        # message donc on considère que chacun à envoyé un message égal à 1 )
        for vertex, neighbours in self.fg.factors_neighbours.items():
            for neighbour in neighbours:
                self.messages[(vertex,neighbour)] = gum.Potential()
        for vertex, neighbours in self.fg.variables_neighbours.items():
            for neighbour in neighbours:
                self.messages[(vertex,neighbour)] = gum.Potential()

        # print("\nOrdre d'envoi :",self.order)

        # Envoi de messages dans tous les sens dans l'ordre défini
        schedule = self.order
        # print(schedule)
        new_message = True # booléen permettant de savoir si de nouveaux messages sont
                           # envoyés/reçus ou non, si plus aucun nouveau message n'est
                           # envoyé cela signifie que l'algorithme a convergé 
        max_nb_iterations = 0 # nombre maximum d'itérations que l'on fixe arbitrairement,
                              # modifiable
        while new_message and max_nb_iterations < 1000: 
            new_message = False
            for vertex in schedule:
                if vertex in self.fg.variables:
                    # print(vertex,"envoie son message à")
                    new_message = send_variable_message(vertex) or new_message
                elif vertex in self.fg.factors:
                    # print(vertex,"envoie son message à")
                    new_message = send_factor_message(vertex)  or new_message
            max_nb_iterations += 1
        
    def posterior(self, v_name):
        """
        Fonction permettant de déterminer la distribution de la variable v_name 
        Entrée:
            v_name : nom de la variable dont on souhaite connaître la distribution
        Sortie:
            posterior : distribution de la variable v_name
        """
        # On fait le produit de tous les messages reçus par v_name ( messages calculés dans
        # makeInference()) puis on normalise le produit obtenu
        posterior = gum.Potential()
        for (sender,receiver), message in self.messages.items():
            if receiver == v_name:
                posterior *= message
        return posterior.normalize()

    def addEvidence(self, evidence):
        """
        Fonction permettant de rajouter des observations sur une ou plusieurs 
        variables dans le Factor Graph fg
        Entrée :
            evidence : dictionnaire d'observations sur une ou plusieurs variables
            { variable X : valeur observée pour X }
        """
        self.fg.addEvidence(evidence)

# Inférence Loopy Belief Propagation Max-Sum

class LBPMaxSumInference:
    
    def __init__(self, fg):
        self.fg = fg.copy() # on travaille avec une copie du Factor Graph pour ne pas
                            # l'altérer
        self.messages = {} # dictionnaire dans lequel on stocke l'ensemble des messages sous
                           # la forme { (émetteur, récepteur) : message }
        self.phi = {} # dictionnaire stockant, pour chaque modalité de chaque variable, 
                      # la valeur la plus probable des variables voisines ( avec lesquelles 
                      # elle a un facteur commun ) pour la probabilité jointe ( produit des
                      # messages qu'elle a reçu )
                      # Le dictionnaire prend la forme suivante :
                      # { ( variable X, valeur de X ) : { variable voisine X_n : valeur 
                      # la plus probable de X_n pour la probabilité jointe quand X = valeur
                      # de X } }

    def makeInference(self):
        """
        Fonction effectuant le calcul de tous les messages ( algorithme Loopy Belief Propagation Max-Sum )
        """
        def send_variable_message(x_m):
            new_message = False
            # x_m envoie un message à chacun de ses voisins
            x_m_neighbours = self.fg.variables_neighbours[x_m]
            for f_s in x_m_neighbours:
                # print(f_s)
                neighbours = [f for f in self.fg.variables_neighbours[x_m] if f != f_s] # voisins de x_m différents de f_s
                # on fait la sommme de tous les messages ( log probas ) venant de chaque
                # voisin de x_m différent de f_s
                message = gum.Potential().fillWith(0)
                for f_n in neighbours: # f_n : facteur voisin
                    message += gum.Potential(self.messages[(f_n,x_m)])
                # print("Avant")
                # gnb.showPotential(self.messages[(x_m,f_s)])
                if self.messages[(x_m,f_s)] != message:
                    new_message = True
                    self.messages[(x_m,f_s)] = message
                    # print("Après")
                    # gnb.showPotential(self.messages[(x_m,f_s)])
            return new_message

        def send_factor_message(f_s):
            new_message = False
            # f_s envoie un message à chacun de ses voisins
            f_s_neighbours = self.fg.factors_neighbours[f_s]
            for x in f_s_neighbours:
                # print(x)
                neighbours = [v for v in self.fg.factors_neighbours[f_s] if v != x] # voisins de f_s différents de x
                # on instancie le message comme étant le log de la table de probabilité 
                # conditionnelle contenue dans f_s
                message = gum.Potential(self.fg.factors_names[f_s])
                for i in message.loopIn():
                    if message.get(i) != 0:
                        message.set(i,np.log(message.get(i))) # on passe au log ( ln )
                    else:
                        message.set(i,-1000000000)  # étant donné que ln(x) tend vers 
                                                    # - l'infini lorsque x tend vers 0,
                                                    # on remplace ici 0 par -1000000000
                                                    # Remarque : une valeur encore plus
                                                    # petite aurait pu être choisie
                if neighbours:
                    # on fait la sommme de tous les messages venant de chaque voisin
                    # de f_s différent de x ( log probas )
                    for x_n in neighbours: # x_n : variable voisine
                        message += gum.Potential(self.messages[(x_n,f_s)])
                # on passe à des valeurs positives pour que Potential.argmax() retourne le
                # bon dictionnaire
                message_abs = gum.Potential(message)
                min_cpt = message_abs.min()
                if min_cpt <= 0:
                    for i in message_abs.loopIn():
                        message_abs.set(i,message_abs.get(i) - min_cpt)
                message = message_abs
                # on stocke dans phi la configuration des variables différentes de x 
                # ( parmi ses variables voisines i.e. séparées d'elle par un facteur ) 
                # ayant donné le maximum pour chaque modalité de x
                if len(message.variablesSequence()) > 1: 
                    I = gum.Instantiation()
                    for v in message.variablesSequence():
                        if v.name() == x:
                            I.add(v)
                            for x_value in list(map(int,v.domain()[1:-1].split(','))):
                                I[x] = x_value
                                phi = message.extract(I).argmax() # configuration des 
                                                                  # variables différentes de
                                                                  # x ayant donné le maximum
                                                                  # pour x = x_value
                                if (x,x_value) in self.phi:
                                    for name, value in phi[0].items():
                                        self.phi[(x,x_value)][name] = value
                                else:
                                    self.phi[(x,x_value)] = phi[0]
                # on fait un max sur toutes les variables de f_s différentes de x
                message = message.margMaxIn([x])
                # print("Avant")
                # gnb.showPotential(self.messages[(x_m,f_s)])
                if self.messages[(f_s,x)] != message:
                    new_message = True
                    self.messages[(f_s,x)] = message
                    # print("Après")
                    # gnb.showPotential(self.messages[(f_s,x)])
            return new_message

        # On détermine les feuilles du Factor Graph s'il en a
        self.leaves = []
        for vertex, neighbours in self.fg.factors_neighbours.items():
            if len(neighbours) == 1:
                self.leaves.append(vertex)
        for vertex, neighbours in self.fg.variables_neighbours.items():
            if len(neighbours) == 1:
                self.leaves.append(vertex)

        # On définit le début de l'ordre d'envoi des messages ( sommets de degré croissant
        # en décrémentant le degré d'un sommet dont un voisin apparaît dans l'ordre )
        # Si le Factor Graph est un arbre, tous les sommets seront ajouter à self.order
        # durant cette étape
        vertices = self.fg.variables + self.fg.factors
        nb_neighbours = [len(self.fg.variables_neighbours[variable]) for variable in self.fg.variables] # nombre de voisins dont on n'a pas encore recu de message
        nb_neighbours += [len(self.fg.factors_neighbours[factor]) for factor in self.fg.factors]
        self.order = []
        self.root = None
        cpt = 0 # compteur : si on ne trouve plus de variable avec un unique voisin, on 
                # arrête
        while len(self.order) < len(vertices):
            new_cpt = cpt
            cpt = 0
            if len(self.order) == len(vertices)-1:
                for i in range(len(vertices)):
                    if vertices[i] not in self.order:
                        self.order.append(vertices[i])
            for i in range(len(vertices)):
                if len(self.order) == len(vertices)-2:
                    for i in range(len(vertices)):
                        if vertices[i] in self.fg.factors and vertices[i] not in self.order:
                            self.order.append(vertices[i])
                if nb_neighbours[i] == 1 and vertices[i] not in self.order:
                    self.order.append(vertices[i]);
                    nb_neighbours[i] = 0
                    for j in range(len(vertices)):
                        if vertices[i] in self.fg.variables and vertices[j] in self.fg.variables_neighbours[vertices[i]] or vertices[i] in self.fg.factors and vertices[j] in self.fg.factors_neighbours[vertices[i]]:
                            nb_neighbours[j] -= 1
                else:
                    cpt += 1
            if new_cpt == cpt and len(self.order) < len(vertices)-1:
                break
        not_in_order = [vertex for vertex in vertices if vertex not in self.order]

        # S'il reste des sommets n'ayant pas été ajouté dans self.order cela signifie que
        # fg n'est pas un arbre
        if not_in_order:
            # on choisit alors une racine au hasard parmi les variables restantes
            self.root = random.choice([vertex for vertex in not_in_order if vertex in self.fg.variables])
            # on détermine un chemin de chacun des sommets restants à cette racine
            self.path_to_root = {}
            self.path_to_root[self.root] = None # parcours sous forme d'un dictionnaire { noeud n : père de n }
            Q = [self.root] # file FIFO ( on enfile un sommet lorsqu'il est découvert et on le défile lorsqu'on a terminé de le traiter )
            while Q:
                u = Q.pop(0)
                if u in self.fg.variables:
                    neighbours = self.fg.variables_neighbours[u]
                elif u in self.fg.factors:
                    neighbours = self.fg.factors_neighbours[u]
                for v in neighbours:
                    if v not in self.path_to_root:
                        self.path_to_root[v]=u
                        Q.append(v)

            # on détermine les feuilles du sous graphe ne contenant que les sommets n'ayant
            # pas encore été ajoutés dans self.order 
            leaves_not_in_order = []
            for vertex in not_in_order:
                if vertex in self.fg.variables:
                    for v,parent in self.path_to_root.items():
                        if parent == vertex and v in self.order:
                            if vertex not in leaves_not_in_order:
                                leaves_not_in_order.append(vertex)

            # on ajoute ensuite les sommets restants des feuilles du sous graphe jusqu'à
            # la racine
            while len(self.order) < len(vertices):
                if leaves_not_in_order:
                    vertex = leaves_not_in_order.pop()
                    while vertex != None and vertex not in self.order and vertex != self.root:
                        self.order.append(vertex)
                        vertex = self.path_to_root[vertex]
                else:
                    if len(self.order) == len(vertices) - 1:
                        self.order.append(self.root)
                    for vertex in vertices:
                        if vertex not in self.order and vertex != self.root:
                            self.order.append(vertex)

        self.root = self.order[-1]

        #print("\nOrdre d'envoi :",self.order)
        #print("\nRacine:",self.root)

        # Initialisation des messages ( chaque sommet doit être en capacité d'envoyer un 
        # message donc on considère que chacun à envoyé un message égal à 1 )
        for vertex, neighbours in self.fg.factors_neighbours.items():
            for neighbour in neighbours:
                self.messages[(vertex,neighbour)] = gum.Potential().fillWith(0)
        for vertex, neighbours in self.fg.variables_neighbours.items():
            for neighbour in neighbours:
                self.messages[(vertex,neighbour)] = gum.Potential().fillWith(0)

        # print("\nOrdre d'envoi :",self.order)
        #print("\nRacine:",self.root)

        # Envoi de messages dans tous les sens dans l'ordre défini
        schedule = self.order
        #print(schedule)
        new_message = True # booléen permettant de savoir si de nouveaux messages sont
                           # envoyés/reçus ou non, si plus aucun nouveau message n'est
                           # envoyé cela signifie que l'algorithme a convergé 
        max_nb_iterations = 0 # nombre maximum d'itérations que l'on fixe arbitrairement,
                              # modifiable
        while new_message and max_nb_iterations < 20: 
            new_message = False
            for vertex in schedule:
                if vertex in self.fg.variables:
                    # print(vertex,"envoie son message à")
                    new_message = send_variable_message(vertex) or new_message
                elif vertex in self.fg.factors:
                    # print(vertex,"envoie son message à")
                    new_message = send_factor_message(vertex) or new_message
            max_nb_iterations += 1
        
    def argmax(self):
        """
        Fonction permettant de déterminer la valeur la plus probable de chaque variable
        du Factor Graph pour la probabilité jointe
        Sortie :
            argmax : dictionnaire des valeurs des variables les plus probables pour la 
            probabilité jointe
        """
        argmax = {}
        if self.root: # si makeInference() a fonctionné correctement, une racine a été
                      # définie
            # on commence par déterminer la valeur la plus probable de la racine en calculant
            # l'argmax de la somme des messages qu'elle a reçu
            m_r = gum.Potential().fillWith(0)
            compute_phi = gum.Potential().fillWith(0)
            for (sender,receiver), message in self.messages.items():
                if receiver == self.root:
                    m_r += message
            argmax = m_r.argmax()[0]

            #print(self.phi)

            # Back-tracking de la racine aux feuilles pour récupérer la valeur la plus 
            # probable de chaque variable
            order = [vertex for vertex in list(reversed(self.order)) if vertex in self.fg.variables and vertex != self.root]
            parent = self.root
            for vertex in order:
                for tpl in self.phi:
                    if tpl[0] == parent and parent in argmax and tpl[1] == argmax[parent]:
                        for v_name, v_value in self.phi[tpl].items():
                            argmax[v_name] = v_value
                parent = vertex

        return argmax

    def addEvidence(self, evidence):
        """
        Fonction permettant de rajouter des observations sur une ou plusieurs 
        variables dans le Factor Graph fg
        Entrée :
            evidence : dictionnaire d'observations sur une ou plusieurs variables
            { variable X : valeur observée pour X }
        """
        self.fg.addEvidence(evidence)


# Low Density Parity Check

def buildLDPC(bits,parity):
    """
    Fonction permettant de construire un réseau bayésien représentant un code LDPC
    Entrées :
        bits : liste de noms de bits ( ex : ['x1','x2','x3'] )
        parity : dictionnaire indiquant les contraintes de parité ( ex : 
        {'pc1':['x1','x2'],'pc2':['x1','x2','x3']} )
    Sortie :
        ldpc : réseau bayésien représentant le code LDPC donné en entrée
    """
    s = ''
    for parity_constraint_name in parity:
        for bit_name in parity[parity_constraint_name]:
            s +=  bit_name + '[0,1]->' +  parity_constraint_name + "[0,1];"
    ldpc = gum.fastBN(s)
    for parity_constraint_name in parity:
        function = ''
        for bit_name in parity[parity_constraint_name]:
            function += 'int('+ bit_name + ')^' # XOR
        function = function[:-1]
        ldpc.cpt(parity_constraint_name).fillWithFunction(function)
    return ldpc

def CBE(bn,message):
    """
    Fonction permettant de reconstruire un dictionnaire des bits manquants
    dans un message grâce à un réseau bayésien donné
    CBE = Canal Binaire à Effacement
    Entrées :
        bn : réseau bayésien permettant de faire des inférences
        message : message incomplet à reconstruire
    Sortie :
        complete_message : message reconstruit
    """
    ie=gum.LazyPropagation(bn)
    ie.setEvidence(message)
    ie.makeInference()
    complete_message = {}
    for i in bn.nodes():
        if bn.variable(i).name() not in message:
            for variable, map_value in ie.posterior(i).argmax()[0].items():
                complete_message[variable] = map_value
    return complete_message

def NoisyLDPC(ldpc,p):
    """
    Fonction permettant de construire un réseau bayésien représentant un
    réseau bayésien dans lequel chaque bit du code LDPC donné apparait 
    dans sa version 'réelle' ( x ) et sa version bruitée ( x' )
    Entrées :
        ldpc : réseau bayésien représentant un LDPC
        p : probabilité d'erreur ( = probabilité que le bit reçu soit
        différent du bit envoyé )
    Sortie :
        nldpc : Noisy LDPC construit
    """
    nldpc = gum.BayesNet(ldpc)
    for i in nldpc.nodes():
        v_name = nldpc.variable(i).name() # bits envoyés
        new_v_name = v_name+"'" # bits reçus
        nldpc.add(new_v_name,2)
        nldpc.addArc(v_name,new_v_name)
        nldpc.cpt(new_v_name).fillWith([1-p,p,p,1-p])
    return nldpc

def CBS(nldpc,message):
    """
    Fonction permettant de reconstruire le message envoyé le plus probable, connaissant
    le message reçu, à l'aide d'un Noisy LDPC
    CBS = Canal Binaire Symétrique
    Entrées :
        nldpc : réseau bayésien représentant un Noisy LDPC
        message : message reçu entier ou incomplet
    Sortie :
        mp_message : message envoyé le plus probable
    """
    ie=gum.LazyPropagation(nldpc)
    ie.setEvidence(message)
    ie.makeInference()
    reconstruction = {}
    for i in nldpc.nodes():
        if nldpc.variable(i).name() not in message:
            for variable, map_value in ie.posterior(i).argmax()[0].items():
                reconstruction[variable] = map_value
    mp_message = {}
    for x, value in reconstruction.items():
        if "'" not in x:
            mp_message[x] = value
    return mp_message