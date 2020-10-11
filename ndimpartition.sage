#from sage.all import *
from sage.structure.list_clone import ClonableArray
#from sage.structure.parent import Parent

class NDimensionalPartition(ClonableArray):
    @staticmethod
    def __classcall_private__(cls, lst):
        """
        Construct an N-dimensional partition with the appropriate parent.
        EXAMPLES::
            sage: PP = NDimensionalPartition([[4,3,3,1], [2,1,1], [1,1]])
            sage: PP.parent() is NDimensionalPartitions(2,17)
            True
        """
        addemup = lambda x: sum(map(addemup, x)) if isinstance(x, list) else x
        n = addemup(lst)
        depth = lambda L : isinstance(L,list) and depth(L[0]) + 1
        N = depth(lst)
        return NDimensionalPartitions(N,n)(lst)

    @staticmethod
    def _list_outside_corners(lst):
        """
        Contructs the outside corners of a N-dimensional partition array as a list of coordinate tuples
        EXAMPLES::
        sage: _list_outside_corners([[5,2,2,1],[3,1,1],[1],[1]])
        []
        """
        depth = lambda L : isinstance(L,list) and depth(L[0]) + 1
        N = depth(lst)
        if lst == []:
            return [(0,)]
        elif lst in ZZ:
            return [(lst,)]
        else:
            out = []
            l = len(lst)
            for i in range(l):
                cocorners = NDimensionalPartition._list_outside_corners(lst[i])
                inquad = (lambda x ,y : reduce (lambda a,b: a and b, map(lambda t: t >= 0, [a - b for (a,b) in zip(x,y)])))
                inquads = (lambda c, checks: reduce(lambda a,b: a or b, [inquad(c,ch) for ch in checks], False))
                new_corners = [c for c in cocorners if not inquads((i,) + c,out)]
                out.extend([(i,) + t for t in new_corners])
            out.append((l,)+(0,)*(N))
            return(out)

    @staticmethod
    def _list_outside_rim(lst):
        if lst == []:
            return [(0,)]
        elif lst in ZZ:
            return [(lst,)]
        else:
            prev_cells = NDimensionalPartition._list_cells(lst[0])
            prev = NDimensionalPartition._list_outside_rim(lst[0])
            out = [(0,) + t for t in prev]
            curr = []
            cur_cells = []
            extras = set()
            l = len(lst)
            for i in range(1,l):
                curr = NDimensionalPartition._list_outside_rim(lst[i])
                curr_cells = NDimensionalPartition._list_cells(lst[i])
                extras = set(prev_cells) - set(curr_cells)
                out.extend([(i,)+ t for t in list(extras.union(set(curr)))])
                prev = curr
                prev_cells = curr_cells
            out.extend([(l,) + t for t in prev_cells])
            return(out)

    def outside_rim(self):
        """
        INPUT: N-dim part NDP
        OUTPUT: the list of cells which share a common face with a cell of NDP but are not in NDP
        """
        return(NDimensionalPartition._list_outside_rim(list(self)))

    @staticmethod
    def _list_cells(lst):
        if lst == []:
            return [(0,)]
        elif lst in ZZ:
            return [(i,) for i in range(lst)]
        else:
            out = []
            l = len(lst)
            for i in range(l):
                out.extend([(i,) + t for t in NDimensionalPartition._list_cells(lst[i])])
            return(out)

    def cells(self):
        return(NDimensionalPartition._list_cells(list(self)))

    @staticmethod
    def _list_from_cells(lst):
        if lst == []:
            return []
        elif len(lst[0]) == 1:
            return max([a+1 for (a,) in lst])
        else:
            piles = {}
            l = 0
            for pt in lst:
                l = max(l, pt[0])
                if pt[0] in piles:
                    piles[pt[0]].append(pt[1:])
                else:
                    piles[pt[0]] = [pt[1:]]
            return [NDimensionalPartition._list_from_cells(piles[i]) for i in range(l+1)]
                
    
    @staticmethod
    def _list_dot_plot(lst, color = 'black'):
        N = len(lst[0])-1
        print "N = %s" % N
        if N > 3 or N == 0:
            raise ValueError("N > 3 requires higher dimensional eyes, N = 0 is stupid")
        if N == 1:
            G = Graphics()
            for c in lst:
                G += circle(c, 0.4, fill = true, color = color, facecolor = color)
            return(G)
        if N == 2:
            G = Graphics()
            for c in lst:
                G += sphere(c, 0.4, color = color, facecolor = color)
            return(G)

    def dot_plot(self, color='black'):
        if self.dimension() > 3:
            raise ValueError("N > 3 requires higher dimensional eyes")
        if self.dimension() == 1:
            cells = self.cells()
            G = Graphics()
            for c in cells:
                G += circle(c, 0.4, fill = true, color = color, facecolor = color)
            return(G)
        if self.dimension() ==2:
            cells = self.cells()
            G = Graphics()
            for c in cells:
                G+= sphere(c, 0.4, color = color, facecolor = color)
            return(G)

    def __init__(self, parent, lst, check=True):
        #depth = lambda L : isinstance(L,list) and depth(L[0]) + 1
        #%N = depth(lst)
        #addemup = lambda x: sum(map(addemup, x)) if isinstance(x, list) else x
        #n = addemup(lst)
        ClonableArray.__init__(self, parent, lst, check)

    def check(self):
        """checks if valid N-dimensional partition
        required for cloneable array class
        really we just check that the tableau is non-increasing
        """
        return(True)

    def __repf__(self):
        return "{}-dimensional partition {}".format(self._N, list(self))

    def next(self):
        """
        Returns the N-dimensional partition of n after self in legicographic order where a
        N-dimensional partition of A is larger than an N-dimensional partition of B if A > B
        EXAMPLES::
            sage: _next(NDimensionalPartition([4,1]))
            [3,2]
            sage: next(NDimensionalPartition([[1,1],[1])
            [[1],[1],[1]]
            sage: next(NDimensionalPartition([[1],[1],[1]])
            False
        """
        if dimension(self) == 1:
            return(NDimensionalPartition(self.parent(),list(Partition(self).next())))
        return(False)

    def _next_inside(self,other = None):
        """
        Helper function for next.
        Returns the next partition of self in lexicographic order subject to the contstraint that
        the next partition must be within other.

        EXAMPLES::
            sage: _next_inside(NDimensionalPartition([4,1]), NDimensionalPartition([2,2,2,2]))
            [2,2,1]
            sage: next_inside(NDimensionalPartition([4,1]), NDimensionalPartition([8,1]))
            False
        """
        if other == None:
            return(next(self))
        else:
            return None
    def size(self):
        return(self.parent().n)
    def dimension(self):
        return(self.parent().N)
    def outside_corners(self):
        """
        Returns a list of the positions where we can add a cell so that the
        shape is still a partition.
        """
        return(NDimensionalPartition._list_outside_corners(list(self)))
    def character(self, vars):
        """
        Assuming the partition is a weight diagram for a module with an action of a torus having generators vars, returns 
        the character of the torus action.
        """
        if len(vars) != self.dimension():
            raise ValueError("the number of variables must equal the dimension of the partition")
        else:
            multipower = lambda a,b : product(x^y for (x,y) in zip(a,b))
            weight_exponents = self.cells()
            weights = map(lambda a : multipower(vars,a), weight_exponents)
            return(sum(weights))
    def monomial_ideal(self, vars):
        """
        Returns the monomial ideal associated with the partition
        """
        if len(vars) != self.dimension():
            raise ValueError("the number of variables must equal the dimension of the partition")
        else:
            multipower = lambda a,b : product(x^y for (x,y) in zip(a,b))
            generator_exponents = self.outside_corners()
            generators = map(lambda a : multipower(vars,a), generator_exponents)
            return(ideal(generators))
    
class NDimensionalPartitions(Parent):
    def __init__(self, N, n):
        self.N = N
        self.n = n
        Parent.__init__(self, category=FiniteEnumeratedSets())


    def __iter__(self):
        for p in self._list_iterator():
            yield self(p)

    def __contains__(self, x):
        """TO DO """
        return(False)

    def __repr__(self):
        return "%s dimensional of partitions of %s"%(self.N, self.n)
    
    def cardinality(self):
        return 0

    @staticmethod
    def largest_inside(lst, k):
        """helper function for iterator.
        INPUT: lst, array representation of N-dim partition
                k, an integer
        OUTPUT: largest N-dim partition of k which is Young-smaller than lst (i.e. fits inside)
        """
        #base case, 0d partition
        if lst in ZZ:
            if k> lst:
                raise ValueError, "k (=%s) cannot be larger than the size of lst"%k
            else:
                return k

        addemup = lambda x: sum(map(addemup, x)) if isinstance(x, list) else x
        wts = map(addemup, lst)

        n = sum(wts)
        if k == n:
            return lst
        elif k> n:
            raise ValueError, "k (=%s) cannot be larger than the size of lst"%k
        else:
            out = []
            l = k
            for i in range(len(wts)):
                if l >= wts[i]:
                    out.append(lst[i])
                    l -= wts[i]
                elif l == 0:
                    break
                else:
                    out.append(NDimensionalPartitions.largest_inside(lst[i], l))
                    break
        return out

    @staticmethod
    def smallest_inside(lst, k):
        """
        helper function for iterator.
        INPUT: lst, array representation of N-dim partition
                k, an integer
        OUTPUT: smallest N-dim partition of k which is Young-smaller than lst (i.e. fits inside)
        """
        #base case, 0d partition
        if isinstance(lst, int):
            if k> lst:
                raise ValueError, "k (=%s) cannot be larger than the size of lst"%k
            else:
                return k
        else:
            transpose = (lambda l : NDimensionalPartition._list_from_cells(map(lambda t : t[::-1], NDimensionalPartition._list_cells(l))))
            return transpose(NDimensionalPartitions.largest_inside(transpose(lst), k))
    
    def first(self):
        if self.N ==1:
            return self.n
        entomb = (lambda n, x: x if n == 0 else [entomb(n-1,x)])
        return self(entomb(self.N-1, 1)*(self.n))
    def last(self):
        if self.N ==1:
            return self.n
        entomb = (lambda n, x: x if n == 0 else [entomb(n-1,x)])
        return self(entomb(self.N - 1, self.n))
        
    
    @staticmethod
    def N_intersect(lss, lst):
        """
        INPUT: two N-dimensional array lists lss lst
        OUTPUT: N-dimensional array list which is the Young-intersection of the two corresponding N-dimensional partitions
        i.e. intersection of the two diagrams.
        """
        if lss in ZZ and lst in ZZ:
            return min(lss, lst)
        else:
            return [NDimensionalPartitions.N_intersect(a,b) for (a,b) in zip(lss, lst)]
    
    @staticmethod
    def _trivial_bound(N,k):
        if N == 1:
            return k
        else:
            explode = lambda a,x : x if a == 1 else [(explode(a-1, x))]*x
            return explode(N, k)
        
    
    def _list_iterator(self, bound = None, boundsize = None):
        """iterator using lists instead of partitions"""
        b = NDimensionalPartitions._trivial_bound(self.N, self.n) if bound == None else bound
        bs = (self.n)**(self.N) if boundsize == None else boundsize
        addemup = lambda x: sum(map(addemup, x)) if isinstance(x, list) else x
        if bs != addemup(b):
            print "bs = %s, not equal size b = %s" %(bs,b)
            print "N = %s, n = %s" %(self.N, self.n)
        if self.n < 0 or self.n > bs:
            return
        elif self.n == bs:
            yield b
        elif self.n == 0:
            yield []
        elif b == []: #works since n != 0
            yield
        elif self.N ==1:
            yield self.n
            return
        else:
            bs0 = addemup(b[0])
            for i in reversed(range(1,self.n+1)):
                for leading_copartition in NDimensionalPartitions(self.N-1, i)._list_iterator(bound = b[0], boundsize = bs0):
                    btail = map((lambda b_i: NDimensionalPartitions.N_intersect(b_i, leading_copartition)), b[1:])
                    btails = addemup(btail)
                    for tail in NDimensionalPartitions(self.N, self.n-i)._list_iterator(bound = btail, boundsize = btails):
                        yield [leading_copartition] + tail

    def _an_element_(self):
        """
        Returns a partition in ``self``.
            TODO
        """
        if self.n == 0:
            lst = []
        elif self.n == 1:
            lst = [1]
        else:
            lst = [self.n-1, 1]
        return self._element_constructor_(lst)

    Element = NDimensionalPartition
