#ifndef CANDQUADTREE_H
#define CANDQUADTREE_H

#include <pthread.h>

#include "cuda_utils.h"

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

#define   VAL_ADDED               (1<<1)
#define   VAL_BELOW               (1<<2)
#define   VAL_ABOVE               (1<<3)

#define   ACTUAL_CONTAINER        (1<<10)
#define   OPTIMISED_CONTAINER     (1<<11)
#define   LMAX_CONTAINER          (1<<12)

#define   NEW_HEAD                (1<<15)
#define   NEW_TAIL                (1<<16)

#define   REMOVE_CONTAINER        (1<<20)
#define   EDITED_CONTAINER        (1<<21)


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

    container*    larger;
    container*    smaller;

    void*         data;

    valNode*      vParent;
    candQuadNode* qParent;

    uint          flag;

    container()
    {
      memset(this, 0, sizeof(container));
    };

    container(initCand* canidate)
    {
      memset(this, 0, sizeof(container));

      data  = new initCand;
      *(initCand*)data = *canidate;
      x     = canidate->r;
      y     = canidate->z;
      val   = canidate->sig;
    };

    ~container()
    {
      freeNull(data);
    }
};

inline container* contFromCand(initCand* canidate)
{
  container* cont = new container(canidate);

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
      memset(this, 0, sizeof(candQuadNode));
    }
};

struct valNode
{
    double      vMin;
    double      vMax;

    container*  min;
    container*  max;

    valNode*    parent;

    uint        noEls;

    valNode*    children[NO_VAL_CHILD];

    valNode()
    {
      memset(this, 0, sizeof(valNode));
    }
};

inline bool operator< ( const initCand cnd, const candQuadNode node )
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


inline  bool operator!= ( const container& cont1, const container& cont2 )
                        {
  if ( ( cont1.val == cont2.val ) && ( cont1.x == cont2.x ) && ( cont1.y == cont2.y ) )
    return false;
  else
    return true;
                        };

inline  bool operator== ( const container& cont1, const container& cont2 )
{
  if ( ( cont1.val == cont2.val ) && ( cont1.x == cont2.x ) && ( cont1.y == cont2.y ) )
    return true;
  else
    return false;
};

inline  bool operator== ( const initCand& cnd, const container& cont )
{
  if ( ( cnd.sig == cont.val ) && ( cnd.r == cont.x ) && ( cnd.z == cont.y ) )
    return true;
  else
    return false;
};


// Intersection of two rectangles
inline bool operator& ( const boundingBox& bb, const candQuadNode& quad )
                                                {
  if ( bb.xMin <= quad.xMax && bb.xMax >= quad.xMin && bb.yMin <= quad.yMax && bb.yMax >= quad.yMin)
  {
    return true;
  }
  else
  {
    return false;
  }
                                                };

// Intersection of point and section
inline bool operator& ( const double& val, const valNode& node )
                                            {
  if ( val >= node.vMin && val <= node.vMax )
    return true;
  else
    return false;
                                            };


inline  double operator| ( const initCand& cnd1, const initCand& cnd2 )
                                                                                                        {
  return ( (cnd1.r - cnd2.r)*(cnd1.r - cnd2.r) + (cnd1.z - cnd2.z)*(cnd1.z - cnd2.z) );
                                                                                                        };

inline  double operator| ( const location& loc, const initCand& cnd2 )
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

    candTree ( );

    ~candTree ( );

    container* insert ( initCand* cnd, double minDist = 1, uint flag = 0 );

    container* get(container* cont, double dist = 0.0 );

    container* getAll(container* cont, double dist = 1 );

    /** Get the initial candidate with the largest sigma within a given distance from a given candidate
     *
     * @param cnd
     * @param dist
     * @return
     */
    container* getLargest(initCand* cnd, double dist = 0.0 );

    container* getLargest(container* cont, double dist = 0.0 );

    uint count();

    uint noVals();

    container* getLargest();

    container* getSmallest();

    uint remove( initCand* cnd );

    uint remove( container* cont );

    void check();

    void markForRemoval(container* cont);

    uint removeMarked();

    int add(candTree* other, double minDist = 0);

  private:
    candQuadNode*   qHead;
    valNode*        vHead;
    pthread_mutex_t editMutex;

    container* getLargest(valNode* node);

    container* getSmallest(valNode* node);

    int insertRec ( container* cont, candQuadNode* node, double minDist );

    candQuadNode* createChild(uint child, candQuadNode* node, container* cont );

    container* getNeighbour(double val, valNode* node);

    int insertRec ( container* cont, valNode* node, double minDist );

    container* insertDecreasing(container* head, container* cont);

    uint insertIntoList(container* head, container* tail, container* cont);

    uint split( valNode* node );

    container* getRec( container* cont, candQuadNode* node, double dist = 4 );

    container* getAllRec( container* cont, candQuadNode* node, boundingBox *bb, double dist, container* list = NULL);

    container* getClosestRec( container* cont, candQuadNode* node, boundingBox *bb, double dist, container* closest = NULL);

    container* getLargestRec( container* cont, candQuadNode* node, boundingBox *bb, double dist, container* largest = NULL);

    uint countRec( candQuadNode* node);

    uint getPos( container* cont, candQuadNode* node );

    uint checkVal();

    uint check(valNode* node);

    uint check(candQuadNode* node);

    uint decValCount(valNode* node);

    uint removeActula( container* cont );

    uint removeMarkedA();

    void freeRec(valNode* node);

    void freeRec(candQuadNode* node);
};


#endif // CANDQUADTREE_H
