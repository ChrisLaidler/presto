#ifndef CANDQUADTREE_H
#define CANDQUADTREE_H

//extern "C"
//{
#include "cuda_accel.h"
//}

#define NO_QUAD_CHILD   4
#define NO_VAL_CHILD    2
#define MAX_VAL_LIST    10


#define TREE_LEFT     (0<<0)
#define TREE_RIGHT    (1<<0)
#define TREE_BOT      (0<<1)
#define TREE_TOP      (1<<1)
#define TREE_BACK     (0<<2)
#define TREE_FRONT    (1<<2)
#define IN            (0<<3)
#define OUT           (1<<3)

#define TREE_TOPTREE_LEFT       ( TREE_TOP | TREE_LEFT  )
#define TREE_TOPTREE_RIGHT      ( TREE_TOP | TREE_RIGHT )

#define TREE_BOTTREE_LEFT       ( TREE_BOT | TREE_LEFT  )
#define TREE_BOTTREE_RIGHT      ( TREE_BOT | TREE_RIGHT )

#define TREE_TOPTREE_LEFTTREE_BACK       ( TREE_TOP | TREE_LEFT  | TREE_BACK)
#define TREE_TOPTREE_RIGHTTREE_BACK      ( TREE_TOP | TREE_RIGHT | TREE_BACK)

#define TREE_BOTTREE_LEFTTREE_BACK       ( TREE_BOT | TREE_LEFT  | TREE_BACK)
#define TREE_BOTTREE_RIGHTTREE_BACK      ( TREE_BOT | TREE_RIGHT | TREE_BACK)

#define TREE_TOPTREE_LEFTTREE_FRONT       ( TREE_TOP | TREE_LEFT  | TREE_FRONT)
#define TREE_TOPTREE_RIGHTTREE_FRONT      ( TREE_TOP | TREE_RIGHT | TREE_FRONT)

#define TREE_BOTTREE_LEFTTREE_FRONT       ( TREE_BOT | TREE_LEFT  | TREE_FRONT)
#define TREE_BOTTREE_RIGHTTREE_FRONT      ( TREE_BOT | TREE_RIGHT | TREE_FRONT)

#define NEW_HEAD    (1<<1)
#define NEW_TAIL    (1<<2)

#define   ADDED   1
#define   BELOW   2
#define   ABOVE   3



#define freeNull(pointer) { if (pointer) free ( pointer ); pointer = NULL; }

struct valNode;
struct candQuadNode;

struct location
{
    double      r;
    float       z;
};

struct boundingBox
{
    double      xMin;
    double      xMax;

    double      yMin;
    double      yMax;
};

struct container
{
    double        x;
    double        y;
    float         val;

    container*    up;
    container*    down;

    void*         data;

    valNode*      vParent;
    candQuadNode* qParent;

    container()
    {
      up      = NULL;
      down    = NULL;
      data    = NULL;
      vParent = NULL;
      qParent = NULL;
    };
};

inline container* contFromCand(cand* canidate)
{
  container* cont = new container;

  cont->data = canidate;
  cont->x     = canidate->r;
  cont->y     = canidate->z;
  cont->val   = canidate->sig;

  if ( cont->vParent != NULL )
  {
    printf("ERROR!\n");
  }

  return cont;
};

struct candQuadNode
{
    double      xMin;
    double      xMax;

    double      yMin;
    double      yMax;

    container*  cont;

    candQuadNode*   parent;
    candQuadNode*   children[NO_QUAD_CHILD];

    candQuadNode()
    {
      cont    = NULL;
      parent  = NULL;

      for ( int i = 0; i < NO_QUAD_CHILD;i++ )
        children[i] = NULL;
    }
};

struct valNode
{
    double      vMin;
    double      vMax;
    double      vMid;

    container*  min;
    container*  max;

    valNode*    parent;

    uint        noEls;

    valNode*    children[NO_VAL_CHILD];

    valNode()
    {
      min     = NULL;
      max     = NULL;
      noEls   = 0;
      parent  = NULL;

      for ( int i = 0; i < NO_VAL_CHILD;i++ )
        children[i] = NULL;
    }
};


inline bool operator< ( const cand cnd, const candQuadNode node )
{
  if ( cnd.r >= node.xMin && cnd.r < node.xMax && cnd.z < node.yMax && cnd.z >= node.yMin )
    return true;
  else
    return false;
};

inline bool operator< ( const location& loc, const candQuadNode& node )
{
  if ( loc.r >= node.xMin && loc.r < node.xMax && loc.z < node.yMax && loc.z >= node.yMin )
    return true;
  else
    return false;
};

inline bool operator< ( const container& cont, const candQuadNode& node )
{
  if ( cont.x >= node.xMin && cont.x < node.xMax && cont.y < node.yMax && cont.y >= node.yMin )
    return true;
  else
    return false;
};

inline bool operator< ( const container& cont, const valNode& node )
{
  if ( cont.val >= node.vMin && cont.val < node.vMax )
    return true;
  else
    return false;
};

inline bool operator< ( const container& cont1, const container& cont2 )
{
  if ( cont1.val < cont2.val )
    return true;
  else
    return false;
};

inline bool operator> ( const container& cont1, const container& cont2 )
{
  if ( cont1.val > cont2.val )
    return true;
  else
    return false;
};

inline bool operator< ( const double& val, const container& cont )
{
  return ( val > cont.val );
};

inline bool operator> ( const double& val, const container& cont )
{
  return ( val > cont.val );
};

// Intersection of two rectangles
inline bool operator& ( const boundingBox& bb, const candQuadNode& quad )
{
  if ( bb.xMin <= quad.xMax && bb.xMax >= quad.xMin && bb.yMin <= quad.yMax && bb.yMax >= quad.yMin)
    return true;
  else
    return false;
};

// Intersection of point and section
inline bool operator& ( const double& val, const valNode& node )
{
  if ( val >= node.vMin && val <= node.vMax )
    return true;
  else
    return false;
};



inline  double operator| ( const cand& cnd1, const cand& cnd2 )
                                                            {
  return ( (cnd1.r - cnd2.r)*(cnd1.r - cnd2.r) + (cnd1.z - cnd2.z)*(cnd1.z - cnd2.z) );
                                                            };

inline  double operator| ( const location& loc, const cand& cnd2 )
                                                    {
  return ( (loc.r - cnd2.r)*(loc.r - cnd2.r) + (loc.z - cnd2.z)*(loc.z - cnd2.z) );
                                                    };

inline  double operator| ( const container& cont1, const container& cont2 )
{
  return ( (cont1.x - cont2.x)*(cont1.x - cont2.x) + (cont1.y - cont2.y)*(cont1.y - cont2.y) );
};

inline  double operator| ( const container& cont1, const candQuadNode& node )
{
  double midX = ( node.xMax + node.xMax ) / 2.0 ;
  double midY = ( node.yMax + node.yMax ) / 2.0 ;
  return ( (cont1.x - midX)*(cont1.x - midX) + (cont1.y - midY)*(cont1.y - midY) );
};


class candTree
{
  public:

    candTree ( )
  {
      FOLD // Set up the quad head  .
      {
        qHead = new candQuadNode;

        qHead->cont      = NULL;
        qHead->parent    = NULL;

        for ( int i = 0; i < NO_QUAD_CHILD;i++ )
          qHead->children[i] = NULL;

        qHead->xMax   = 1000;
        qHead->xMin    = 0;

        qHead->yMax   = 100;
        qHead->yMin   = -100;
      }

      FOLD // Set up the value head  .
      {
        vHead = new valNode;

        vHead->vMin     = 0;
        vHead->vMax     = 100;
        vHead->vMid     = ( vHead->vMax + vHead->vMin ) / 2.0 ;

        vHead->min     = NULL;
        vHead->max     = NULL;
        vHead->noEls    = 0;
        vHead->parent   = NULL;

        for ( int i = 0; i < NO_VAL_CHILD;i++ )
          vHead->children[i] = NULL;
      }

      minDist       = 1;
  }

    int insert ( cand* cnd )
    {
      container* cont = contFromCand(cnd);

      printf("\n▶ Insert canidate\n");
      int res = insertRec ( cont, qHead );
      if ( res != BELOW )
        insertRec ( cont, vHead );

      check(vHead);
      check(qHead);
      checkVal();
    }

    cand* get(double r, float z, double dist = 4 )
    {
      container loc;
      loc.x = r;
      loc.y = z;

      container* cont = getRec( &loc, qHead, dist);

      return  (cand*)cont->data;
    }

    container* get(container* cont, double dist = 0.5 )
    {
      boundingBox bb;
      bb.xMin = cont->x - dist/2.0;
      bb.xMax = cont->x + dist/2.0;

      bb.yMin = cont->y - dist/2.0;
      bb.yMax = cont->y + dist/2.0;

      return getClosestRec( cont, qHead, &bb, dist);
    };

    container* getAll(container* cont, double dist = 4 )
    {
      boundingBox bb;
      bb.xMin = cont->x - dist/2.0;
      bb.xMax = cont->x + dist/2.0;

      bb.yMin = cont->y - dist/2.0;
      bb.yMax = cont->y + dist/2.0;

      container* lst = getAllRec( cont, qHead, &bb, dist);
      return lst;
    }

    uint count()
    {
      return countRec(qHead);
    };

    uint noVals()
    {
      return vHead->noEls;
    };

    container* getLargest()
    { return getLargest(vHead);};

    uint remove( cand* cnd )
    {
      container* cont = contFromCand(cnd);
      return remove( cont );
    }

    uint remove( container* cont )
    {
      container* actual = get(cont, 0.0);
      if ( actual )
      {
        return removeActula(actual);
      }
    }

  private:
    ///< The qHead Node
    candQuadNode*   qHead;
    valNode*        vHead;
    double          minDist;

    container* getLargest(valNode* node)
    {
      if ( node->children[1] )
      {
        if ( node->children[1]->noEls )
          return getLargest( node->children[1] );
      }

      if ( node->children[0] )
      {
        if ( node->children[0]->noEls )
          return getLargest( node->children[0] );
      }

      if ( node->max )
        return node->max ;

      return NULL; // It probably shouldest get to this point!

    }

    int insertRec ( container* cont, candQuadNode* node )
    {
      //printf("•\n");

      if ( *cont < *node )
      {
        if    ( node->cont )
        {
          printf("x Node has candidate\n");

          double dist = *cont | *node->cont ;

          if ( dist < minDist )
          {
            printf("O \n");
            if ( cont->val > node->cont->val )
            {
              printf("▲ Candidate above current\n");

              removeActula( node->cont );

              node->cont        = cont;
              cont->qParent     = node;
              return ABOVE;
            }
            else
            {
              // Don't add!
              printf("▼ Candidate below current\n");

              removeActula( cont );

              return BELOW;
            }
          }
          else
          {
            //printf("Not within distance\n");

            uint down = getPos(node->cont, node);
            if ( node->children[down] == NULL ) // Create the child node  .
            {
              printf("⤋ push existing candidate\n");

              createChild(down, node, node->cont);
              node->cont = NULL; // clear candidate of this node
            }
            else
            {
              // This probably shouldn't happen
              // Push the candidate down.

              container* hold   =  node->cont;
              node->cont        = NULL;
              hold->qParent     = NULL;
              insertRec(hold, node->children[down]);
            }

            uint child = getPos(cont, node);
            if ( node->children[child] == NULL ) // Create the child node  .
            {
              printf("↓ Create new child for current candidate\n");

              createChild(child, node, cont);
              return ADDED;
            }
            else
            {
              // Recurs down
              printf("↓ Recurs down\n");
              return insertRec ( cont, node->children[child] );
            }
          }
        }
        else
        {
          bool leaf = true;

          // Check if this is a leaf node
          for ( int i = 0; i < NO_QUAD_CHILD; i++ )
          {
            if ( node->children[i] )
            {
              leaf = false;
              break;
            }
          }

          if ( leaf )
          {
            printf("+ Leaf so add here\n");
            node->cont      = cont;
            cont->qParent   = node;
            return ADDED;
          }
          else
          {
            //printf("Not leaf\n");
            uint child = getPos(cont, node);

            if ( node->children[child] == NULL ) // Create the child node  .
            {
              printf("↓ Create new child for current candidate\n");

              createChild(child, node, cont);
              return ADDED;
            }
            else
            {
              // Recurs down
              printf("↓ Recurs down\n");
              return insertRec ( cont, node->children[child] );
            }
          }
        }
      }
      else
      {
        if ( node->parent )
        {
          // Recurs UP
          printf("↑ Recurs UP\n");
          return insertRec ( cont, node->parent );
        }
        else
        {
          printf("Create new qHead\n");

          node->parent = new candQuadNode;

          node->parent->parent    = NULL;
          node->parent->cont      = NULL;
          node->parent->parent    = NULL;

          for ( int i = 0; i < NO_QUAD_CHILD;i++ )
          {
            node->parent->children[i] = NULL;
          }

          uint child = 0;

          FOLD // Calculate my position in parent and set the parent's bounds  .
          {
            if ( cont->x < node->xMin )
            {
              child |= TREE_RIGHT;
              node->parent->xMin = node->xMin - ( node->xMax - node->xMin );
              node->parent->xMax = node->xMax;
            }
            else
            {
              child |= TREE_LEFT;
              node->parent->xMin = node->xMin;
              node->parent->xMax = node->xMax + ( node->xMax - node->xMin );
            }

            if ( cont->y < node->yMin )
            {
              child |= TREE_TOP;
              node->parent->yMin = node->yMin - ( node->yMax - node->yMin );
              node->parent->yMax = node->yMax;
            }
            else
            {
              child |= TREE_BOT;
              node->parent->yMin = node->yMin;
              node->parent->yMax = node->yMax + ( node->yMax - node->yMin );
            }
          }

          // Set me as a child
          node->parent->children[child] = node;

          // This must be the new qHead node
          qHead = node->parent;

          // Recurs UP
          printf("↑ Recurs UP\n");
          return insertRec ( cont, node->parent );
        }
      }
    }

    candQuadNode* createChild(uint child, candQuadNode* node, container* cont )
    {
      node->children[child] = new candQuadNode;

      for ( int i = 0; i < NO_QUAD_CHILD; i++ )
        node->children[child]->children[i] = NULL;

      node->children[child]->parent = node;
      node->children[child]->cont   = cont;
      cont->qParent = node->children[child];

      FOLD // Set the bounds of the child node
      {
        if ( child & TREE_RIGHT )
        {
          node->children[child]->xMin = node->xMin + ( node->xMax - node->xMin ) / 2.0;
          node->children[child]->xMax = node->xMax;
        }
        else
        {
          node->children[child]->xMin = node->xMin;
          node->children[child]->xMax = node->xMin + ( node->xMax - node->xMin ) / 2.0;
        }

        if ( child & TREE_TOP )
        {
          node->children[child]->yMin = node->yMin + ( node->yMax - node->yMin ) / 2.0;
          node->children[child]->yMax = node->yMax;
        }
        else
        {
          node->children[child]->yMin = node->yMin;
          node->children[child]->yMax = node->yMin + ( node->yMax - node->yMin ) / 2.0;
        }
      }
    }

    container* getNeighbour(double val, valNode* node)
    {
      if ( !(node->noEls) )
        return NULL;

      int order[2];

      double mid = ( node->vMin + node->vMax ) / 2.0 ;

      double d1 = mid - val;
      if ( d1 > 0 )
      {
        order[0] = 0;
        order[1] = 1;
      }
      else
      {
        order[0] = 1;
        order[1] = 0;
      }

      for ( int i = 0; i < NO_VAL_CHILD; i++ )
      {

        if ( node->children[order[i]] )
        {
          container* res;

          res = getNeighbour( val, node->children[order[i]] );

          if ( res )
            return res;
        }
      }

      if ( node->max )
      {
        if ( val & *node )
        {
          container* cand = node->max;

          while ( val > *cand )
          {
            if ( cand->up == NULL )
            {
              return cand;
            }
            cand = cand->up;
          }
          return cand;
        }
        else if ( node->vMax < val )
        {
          container* cand = node->max;

          while ( val > *cand )
          {
            if ( cand->up == NULL )
            {
              return cand;
            }
            cand = cand->up;
          }
          return cand;
        }
        else if ( node->vMin > val )
        {
          container* cand = node->min;

          while ( val < *cand )
          {
            if ( cand->down == NULL )
            {
              return cand;
            }
            cand = cand->down;
          }
          return cand;
        }
      }
      return NULL; // Probably can't get here with the noEls
    }

    int insertRec ( container* cont, valNode* node )
    {
      if ( node->noEls == 0 && !( node->max == NULL && node->max == NULL) )
      {
        printf("We have a problem!\n");
      }

      if ( *cont < *node )
      {
        bool leaf = true;

        for ( int i = 0; i < NO_VAL_CHILD; i++ )
        {
          if ( node->children[i] )
          {
            leaf = false;
          }
        }

        if ( leaf )
        {
          if ( node->noEls < MAX_VAL_LIST )
          {
            if ( node->noEls <= 0 )
            {
              container* close = getNeighbour(cont->val, vHead);

              if ( close )
              {
                if ( *close < *cont )
                {
                  if (  close->up )
                  {
                    if ( !( *cont < *close->up) )
                    {
                      printf("ERROR: value not between bounds\n");
                    }
                  }
                  else
                  {
                    printf("END\n");
                  }

                  container* hold = close->up;
                  close->up = cont;
                  cont->down = close;
                  if ( hold )
                    hold->down = cont;
                  cont->up = hold;
                }
                else
                {
                  if ( close->down )
                  {
                    if ( !( *close->down < *cont) )
                    {
                      printf("ERROR: value not between bounds\n");
                    }
                  }
                  else
                  {
                    printf("Beginning\n");
                  }

                  container* hold = close->down;
                  close->down = cont;
                  cont->up = close;
                  if ( hold )
                    hold->up = cont;
                  cont->down = hold;
                }
              }
            }

            uint res = insert(node->min, node->max, cont);

            cont->vParent = node;

            if ( res & NEW_HEAD )
            {
              //printf("new min\n");
              node->min = cont;
            }

            if ( res & NEW_TAIL )
            {
              //printf("new max\n");
              node->max = cont;
            }

            node->noEls++;

            FOLD // TMP  .
            {
              container* mark = node->min;
              for ( int i = 0; i < node->noEls; i++ )
              {
                //printf("%7.3f ",mark->val);

                if ( i == node->noEls - 1)
                {
                  if ( mark != node->max )
                  {
                    printf("We have a problem!\n");
                  }
                }
                mark = mark->up;
              }
              //printf("\n");

            }

            return 1;
          }
          else  // This node has too many elements split it and add cont to relevant child  .
          {
            split( node );

            for ( int i = 0; i < NO_VAL_CHILD;i++ )
            {
              if ( *cont < *node->children[i] )
              {
                uint res = insertRec( cont, node->children[i] );

                if ( res == 1 )
                  node->noEls++;

                return res;
              }
            }
          }
        }
        else // Not a leaf so investigate existing children  .
        {
          double mid = ( node->vMin + node->vMax ) / 2.0 ;
          int child = 0;

          if ( cont->val < mid )
          {
            child |= TREE_LEFT;
          }
          else
          {
            child |= TREE_RIGHT;
          }

          if ( !(node->children[child]) )
          {
            valNode* down = new valNode;

            if ( child & TREE_LEFT )
              down->vMax = mid;
            else
              down->vMin = mid;

            down->parent = node;

            node->children[child] = down;
          }

          // Recurse down
          uint res = insertRec ( cont, node->children[child] );

          if ( res == 1 )
            node->noEls++;

          return res;
        }
      }
      else // Not in this node recurse up  .
      {
        valNode* up;
        up->noEls = node->noEls;

        if ( cont->val < node->vMin )
        {
          up->vMin = node->vMin - ( node->vMax - node->vMin );
          up->vMax = node->vMin;
        }
        else
        {
          up->vMin = node->vMax;
          up->vMax = node->vMax + ( node->vMax - node->vMin );
        }

        // Recurse up
        uint res = insertRec ( cont, up );

        if ( res == 1 )
          node->noEls++;

        return res;
      }
    }

    container* insertDecreasing(container* head, container* cont)
    {
      if ( head == NULL )
        return cont;

      container* mark = head;

      while ( *mark > *cont )
      {
        if ( mark->down == NULL )
        {
          container* hold =  mark->down;
          mark->down  = cont;
          cont->down  = hold;
          cont->up    = mark;
          if( hold )
            hold->up  = cont;

          return head;
        }
        mark = mark->down;
      }

      container* hold =  mark->up;
      mark->up        = cont;
      cont->up        = hold;
      cont->down      = mark;
      if( hold )
        hold->down    = cont;

      if ( mark == head )
        return cont;
      else
        return head;

    }

    uint insert(container* head, container* tail, container* cont)
    {
      uint res = 0;

      if ( head == NULL && tail == NULL )
      {
        res |= NEW_HEAD;
        res |= NEW_TAIL;

        //cont->down  = NULL;
        //cont->up    = NULL;

        return res;
      }

      container* mark = head;

      while ( *mark < *cont )
      {
        if ( mark == tail )
        {
          res |= NEW_TAIL; // This should always be the case
        }

        if ( mark->up == NULL )
        {
          container* hold =  mark->up;
          mark->up        = cont;
          cont->up        = hold;
          cont->down      = mark;
          if( hold )
            hold->down    = cont;

          return res;
        }
        mark = mark->up;
      }

      if ( mark == head )
      {
        res |= NEW_HEAD;
      }

      container* hold =  mark->down;
      mark->down  = cont;
      cont->down  = hold;
      cont->up    = mark;
      if( hold )
        hold->up  = cont;

      return res;
    }

    uint split( valNode* node )
    {

      //printf("Split  %8.4f - %8.4f  \n", node->vMin, node->vMax );

      //      FOLD // TMP  .
      //      {
      //        container* mark = node->min;
      //        for ( int i = 0; i < node->noEls; i++ )
      //        {
      //          printf("%7.3f ",mark->val);
      //          mark = mark->up;
      //        }
      //        printf("\n");
      //      }


      valNode* left   = new valNode;
      valNode* right  = new valNode;

      node->children[0] = left;
      node->children[1] = right;

      double mid = ( node->vMin + node->vMax ) / 2.0 ;

      left->vMin = node->vMin;
      left->vMax = mid;
      right->vMin = mid;
      right->vMax = node->vMax;

      container* cont = node->min;

      FOLD //  loop through left elements  .
      {
        while ( *cont < *left )
        {
          left->noEls++;
          cont->vParent = left;

          //printf("%7.3f ", cont->val); // TMP

          cont = cont->up;
          if ( cont == NULL )
            break;
        }
      }

      right->noEls  = node->noEls - left->noEls;

      if ( left->noEls == 0 )
      {
        left->min     = NULL;
        left->max     = NULL;

        right->min     = node->min;
        right->max     = node->max;
      }
      else if ( left->noEls < node->noEls )
      {
        left->min       = node->min;
        left->max       = cont->down;

        right->min    = cont      ;
        right->max    = node->max ;

      }
      else
      {
        left->min     = node->min;
        left->max     = node->max;

        right->min    = NULL ;
        right->max    = NULL ;
      }

      //printf("|");

      FOLD // Set right parent  .
      {
        uint cnt = 0;

        cont = right->min;

        while ( cont != right->max )
        {
          cont->vParent = right;
          //printf("%7.3f ", cont->val);

          cont = cont->up;
          cnt++;
        }
        if (cont)
        {
          cont->vParent = right;
          //printf("%7.3f ", cont->val);
          cnt++;
        }
        //printf("\n");

        if ( cnt != right->noEls ) // TMP  .
        {
          printf("We have a problem!\n");

          cnt = 0;

          cont = right->min;

          while ( cont != right->max )
          {
            cont->vParent = right;
            printf("%7.3f ", cont->val);

            cont = cont->up;
            cnt++;
          }
          if (cont)
          {
            cont->vParent = right;
            printf("%7.3f ", cont->val);
            cnt++;
          }
          printf("\n");

        }
      }

      left->parent  = node;
      right->parent = node;

      node->min = NULL;
      node->max = NULL;
    }

    container* getRec( container* cont, candQuadNode* node, double dist = 4 )
    {
      if ( *cont < *node)
      {
        if ( node->cont )
        {
          double cndDist = *cont | *node->cont ;

          if ( cndDist <= dist )
          {
            return node->cont;
          }
          else
          {
            // Just for fun lets check children
            for ( int i = 0; i < NO_QUAD_CHILD; i++ )
            {
              if ( node->children[i] )
              {
                if ( *cont < *node->children[i] )
                {
                  printf("ERROR node with cand has child?");
                  return getRec( cont, node, dist );
                }
              }
            }
          }
        }

      }
      else
      {
        return NULL;
      }
    }

    container* getAllRec( container* cont, candQuadNode* node, boundingBox *bb, double dist, container* list = NULL)
    {
      if (node)
      {
        if ( *bb & *node )
        {
          //printf(" x [ %11.3f %11.3f ] Y [ %11.3f %11.3f ]\n",node->xMin, node->xMax, node->yMax, node->yMax );

          for ( int i = 0; i < NO_QUAD_CHILD; i++ )
          {
            if ( node->children[i] )
            {
              list = getAllRec( cont, node->children[i], bb, dist, list);
            }
          }

          // this is a leaf so search my value!
          if ( node->cont )
          {
            double dst = *cont | *node->cont ;
            if ( dst < dist )
            {
              // Within distance so add to list!

              container* newCont = contFromCand((cand*)node->cont->data);
              list = insertDecreasing(list, newCont);
            }
          }
        }
      }
      return list;
    }

    container* getClosestRec( container* cont, candQuadNode* node, boundingBox *bb, double dist, container* closest = NULL)
    {
      if (node)
      {
        if ( *bb & *node )
        {
          //double tmpDST = *cont | *node ;
          //tmpDST = sqrt(tmpDST);
          //printf(" x [ %11.3f %11.3f ] Y [ %11.3f %11.3f ]   %16.3f\n",node->xMin, node->xMax, node->yMin, node->yMax, tmpDST );

          for ( int i = 0; i < NO_QUAD_CHILD; i++ )
          {
            if ( node->children[i] )
            {
              if ( *bb & *node->children[i] )
              {
                closest = getClosestRec( cont, node->children[i], bb, dist, closest);
              }
            }
          }

          // this is a leaf so search my value!
          if ( node->cont )
          {
            double newDst = *cont | *node->cont ;

            if ( newDst <= dist )
            {
              if (closest)
              {
                double oldDst = *cont | *closest ;
                if ( newDst < oldDst )
                {
                  // Within distance so add to list!
                  closest = node->cont;
                }
              }
              else
              {
                closest =node->cont;
              }
            }
          }
        }
      }
      return closest;
    }

    uint countRec( candQuadNode* node)
    {
      uint cnt = 0;

      if (node->cont)
      {
        cnt++;

#ifndef DEBUG
        for ( int i = 0; i < NO_QUAD_CHILD; i++ )
        {
          if ( node->children[i] )
          {
            printf("Node with cand that is not a leaf!\n");
          }
        }
#endif
      }
      else
      {
        for ( int i = 0; i < NO_QUAD_CHILD; i++ )
        {
          if ( node->children[i] )
          {
            cnt += countRec(node->children[i]);
          }
        }
      }

      return cnt;

    }

    uint getPos( container* cont, candQuadNode* node )
    {
      uint child = 0 ;

      if ( !( *cont < *node) )
      {
        return child;
      }

      FOLD // Find location of child
      {
        if ( cont->x < node->xMin + ( node->xMax - node->xMin ) / 2.0 )
          child |= TREE_LEFT;
        else
          child |= TREE_RIGHT;

        if ( cont->y < node->yMin + ( node->yMax - node->yMin ) / 2.0 )
          child |= TREE_BOT;
        else
          child |= TREE_TOP;
      }

      return child;
    }

    uint checkVal()
    {
      container* cont = getLargest(vHead);

      int cnt = 0;

      while(cont)
      {
        if (cont->down)
        {
          if ( *cont < *cont->down )
          {
            printf("ERROR: values not in order!\n");
          }
        }
        cnt++;
        cont = cont->down;
      }

      if ( cnt != vHead->noEls )
      {
        printf("ERROR: Wrong number of elements!\n");
      }

      int tmp = 0;
    }

    uint check(valNode* node)
    {
      if ( node->min || node->max )
      {
        if ( node->min == NULL || node->max == NULL )
          printf("ERROR: node->min == NULL || node->max == NULL \n");

        container* cont = node->min;
        int cnt = 0;
        FOLD //  loop through left elements  .
        {
          while ( cont != node->max )
          {
            cnt++;
            if ( cont->vParent != node )
              printf("ERROR: cont->vParent != node \n");

            cont = cont->up;
          }
          if ( cont == node->max )
          {
            cnt++;
            if ( cont->vParent != node )
              printf("ERROR: cont->vParent != node \n");

          }

          if ( cnt != node->noEls )
            printf("ERROR: cnt != node->noEls \n");
        }
      }

      for ( int i = 0; i < NO_VAL_CHILD;i++ )
      {
        if ( node->children[i] )
        {
          check(node->children[i]);
        }
      }
    }

    uint check(candQuadNode* node)
    {
      if ( node )
      {
        if ( node->cont )
        {
          if ( node->cont->qParent != node )
            printf("ERROR: node->cont->qParent != node \n");
        }
      }

      for ( int i = 0; i < NO_QUAD_CHILD;i++ )
      {
        if ( node->children[i] )
        {
          check(node->children[i]);
        }
      }
    }

    uint decValCount(valNode* node)
    {
      if ( node )
      {
        node->noEls --;
        decValCount(node->parent);
      }
    }

    uint removeActula( container* cont )
    {
      printf("Remove:  Value: %.4f  X: %.4f  Y: %.4f \n", cont->val, cont->x, cont->y );

      if ( !(cont->up) && !(cont->down) && (cont->vParent) )
        int tmep = 0;

      if ( cont->up )
        cont->up->down = cont->down;

      if ( cont->down )
        cont->down->up = cont->up;

      if ( !(cont->up) && !(cont->down) )
      {
        if ( cont->vParent )
        {
          if ( cont->vParent->min == cont &&  cont->vParent->max == cont )
          {
            cont->vParent->min = NULL;
            cont->vParent->max = NULL;
          }
        }
      }
      else if ( cont->up && cont->down )
      {
        if ( cont->vParent->min == cont->vParent->max )
        {
          cont->vParent->min = NULL;
          cont->vParent->max = NULL;
        }
        else
        {
          if ( cont->up->vParent !=  cont->down->vParent )
          {
            if ( cont->vParent->min == cont )
            {
              if ( cont->up->vParent == cont->vParent )
              {
                cont->vParent->min = cont->up;
              }
              else
              {
                cont->vParent->min = NULL;
                cont->vParent->max = NULL;
              }
            }

            if ( cont->vParent->max == cont )
            {
              if ( cont->down->vParent == cont->vParent )
              {
                cont->vParent->max = cont->down;
              }
              else
              {
                cont->vParent->min = NULL;
                cont->vParent->max = NULL;
              }
            }
          }
        }
      }
      else
      {
        if ( cont->up )
        {
          if ( cont->vParent == cont->up->vParent)
          {
            cont->vParent->min = cont->up ;
          }
          else
          {
            cont->vParent->max = NULL;
            cont->vParent->min = NULL;
          }
        }

        if ( cont->down )
        {
          if ( cont->vParent == cont->down->vParent)
          {
            cont->vParent->max = cont->down ;
          }
          else
          {
            cont->vParent->max = NULL;
            cont->vParent->min = NULL;
          }
        }

      }

      FOLD //  Set parents  .
      {
        if ( cont->vParent )
        {
          decValCount(cont->vParent);

          if ( cont->vParent->noEls == 0 && !(cont->vParent->max == NULL && cont->vParent->max == NULL) ) // TMP  .
          {
            printf("We have a problem!\n");
          }
        }

        if ( cont->qParent )
        {
          cont->qParent->cont   = NULL;
          cont->qParent         = NULL; // Not strictly necessary as we are going to delete this anyway
        }
      }

      freeNull(cont->data);
      freeNull(cont);
    }
};




#endif // CANDQUADTREE_H
