#include "candTree.h"


candTree::candTree ( )
{
  FOLD // Set up the quad head  .
  {
    qHead = new candQuadNode;

    qHead->cont      = NULL;
    qHead->parent    = NULL;

    for ( int i = 0; i < NO_QUAD_CHILD;i++ )
      qHead->children[i] = NULL;

    qHead->xMax     = 1000;
    qHead->xMin     = 0;

    qHead->yMax     = 100;
    qHead->yMin     = -100;
  }

  FOLD // Set up the value head  .
  {
    vHead = new valNode;

    vHead->vMin     = 0;
    vHead->vMax     = 100;

    vHead->min      = NULL;
    vHead->max      = NULL;
    vHead->noEls    = 0;
    vHead->parent   = NULL;

    for ( int i = 0; i < NO_VAL_CHILD;i++ )
      vHead->children[i] = NULL;
  };

  //editMutex = PTHREAD_MUTEX_INITIALIZER ;
  pthread_mutex_init(&editMutex, NULL);

}

candTree::~candTree ( )
{
  pthread_mutex_lock( &editMutex );

  container* cont = getLargest();
  uint cnt = 0;
  container* next = cont;
  while (next)
  {
    cont = next;
    next = next->smaller;
    freeNull(cont->data);
    freeNull(cont);
    cnt++;
  }

  freeRec(qHead);
  freeRec(vHead);

  pthread_mutex_unlock( &editMutex );
}

container* candTree::insert ( cand* cnd, double minDist, uint flag )
{
  minDist = minDist*minDist; // Square cos all measured distances aren't square rooted

  container* cont = new container(cnd);
  cont->flag |= ACTUAL_CONTAINER;

  pthread_mutex_lock( &editMutex );

  //debugMessage("▶ Insert candidate\n");
  int resQ = insertRec ( cont, qHead, minDist );
  int resV = 0;

  //check();

  if ( resQ & VAL_ADDED )
  {
    //debugMessage("-----------------------------------\n");
    resV = insertRec ( cont, vHead, minDist );

    if ( resV != 1 )
    {
      errMsg("ERROR: failed to add value in to tree. Line %i in %s", __LINE__, __FILE__ );
    }
  }

  //check();

  pthread_mutex_unlock( &editMutex );

  if ( (resQ & VAL_ADDED) &&  (resV == 1) )
  {
    return cont;
  }
  else
  {
    removeActula(cont);
    return NULL;
  }
}

container* candTree::get(container* cont, double dist)
{
  boundingBox bb;
  bb.xMin = cont->x - dist;
  bb.xMax = cont->x + dist;

  bb.yMin = cont->y - dist;
  bb.yMax = cont->y + dist;

  dist = dist*dist; // Square cos all measured distances aren't square rooted

  return getClosestRec( cont, qHead, &bb, dist);
};

container* candTree::getAll(container* cont, double dist)
{
  boundingBox bb;
  bb.xMin = cont->x - dist;
  bb.xMax = cont->x + dist;

  bb.yMin = cont->y - dist;
  bb.yMax = cont->y + dist;

  dist = dist*dist; // Square cos all measured distances aren't square rooted

  container* lst = getAllRec( cont, qHead, &bb, dist);
  return lst;
}

container* candTree::getLargest(cand* cnd, double dist )
{
  container* cont  = new container(cnd);

  container* ret = getLargest(cont, dist );

  delete(cont);

  return ret;
}

container* candTree::getLargest(container* cont, double dist )
{
  boundingBox bb;
  bb.xMin = cont->x - dist;
  bb.xMax = cont->x + dist;

  bb.yMin = cont->y - dist;
  bb.yMax = cont->y + dist;

  dist = dist*dist; // Square cos all measured distances aren't square rooted

  return getLargestRec( cont, qHead, &bb, dist);
};

uint candTree::count()
{
  return countRec(qHead);
};

uint candTree::noVals()
{
  return vHead->noEls;
};

container* candTree::getLargest()
{
  return getLargest(vHead);
};

container* candTree::getSmallest()
{
  return getSmallest(vHead);
};

uint candTree::remove( cand* cnd )
{
  container* cont = new container(cnd);
  return remove( cont );
}

uint candTree::remove( container* cont )
{
  uint result = 0;

  pthread_mutex_lock( &editMutex );
  FOLD // Remove
  {
    if ( !(cont->flag & ACTUAL_CONTAINER) )
    {
      // This is a dummy container get the actual one
      cont = get(cont, 0.0);
    }

    if ( cont )
    {
      result =  removeActula(cont);
    }
  }
  pthread_mutex_unlock( &editMutex );

  return result;
}

void candTree::check()
{
  check(vHead);
  //check(qHead);
  checkVal();
}

void candTree::markForRemoval(container* cont)
{
  if ( !(cont->flag & ACTUAL_CONTAINER) )
  {
    // This is a dummy container get the actual one
    cont = get(cont, 0.0);
  }
  cont->flag |= REMOVE_CONTAINER ;
}

uint candTree::removeMarked()
{
  uint result = 0;

  pthread_mutex_lock( &editMutex );
  result = removeMarkedA();
  pthread_mutex_unlock( &editMutex );

  return result;
}

int candTree::add(candTree* other, double minDist)
{
  uint cnt        = 0;

  if ( other )
  {
    minDist = minDist*minDist; // Square cos all measured distances aren't square rooted

    pthread_mutex_lock( &editMutex );

    container* cont = other->getLargest();

    while (cont)
    {
      cand* canidate = (cand*)cont->data;

      container* contNew = new container(canidate);
      cont->flag |= ACTUAL_CONTAINER;

      //printf("\nInsert spatial\n");
      int res = insertRec ( contNew, qHead, minDist );

      if ( res & VAL_ADDED )
      {
        //printf("Insert value\n");
        res = insertRec ( contNew, vHead, minDist );
      }

      cnt++;
      cont = cont->smaller;
    }

    pthread_mutex_unlock( &editMutex );
  }

  return cnt;
}

container* candTree::getLargest(valNode* node)
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
  {
    // Just encase loop
    container* hold = node->max;
    while(hold->larger)
    {
      hold = hold->larger;
    }

    return hold ;
  }

  return NULL; // It probably shouldest get to this point!

}

container* candTree::getSmallest(valNode* node)
{
  if ( node->children[0] )
  {
    if ( node->children[0]->noEls )
      return getSmallest( node->children[0] );
  }

  if ( node->children[1] )
  {
    if ( node->children[1]->noEls )
      return getSmallest( node->children[1] );
  }

  if ( node->min )
  {
    // Just encase loop
    container* hold = node->min;
    while(hold->smaller)
    {
      hold = hold->smaller;
    }

    return hold ;
  }

  return NULL; // It probably shouldest get to this point!

}

int candTree::insertRec ( container* cont, candQuadNode* node, double minDist )
{
  //debugMessage("•\n");

  if ( *cont < *node )
  {
    if    ( node->cont )
    {
      //debugMessage("x Node has candidate\n");

      double dist = *cont | *node->cont ;

      if ( dist < minDist )
      {
        //debugMessage("○\n");
        if ( cont->val > node->cont->val )
        {
          //debugMessage("▲ Candidate above current\n");

          removeActula( node->cont );

          node->cont        = cont;
          cont->qParent     = node;
          return (  VAL_ABOVE | VAL_ADDED );
        }
        else
        {
          // Don't add!
          //debugMessage("▼ Candidate below current\n");

          //removeActula( cont );

          return VAL_BELOW;
        }
      }
      else
      {
        //debugMessage("Not within distance\n");

        uint down = getPos(node->cont, node);
        if ( node->children[down] == NULL ) // Create the child node  .
        {
          //debugMessage("⤋ push existing candidate\n");

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
          insertRec(hold, node->children[down], minDist);
        }

        uint child = getPos(cont, node);
        if ( node->children[child] == NULL ) // Create the child node  .
        {
          //debugMessage("↓ Create new child for current candidate\n");

          createChild(child, node, cont);
          return VAL_ADDED;
        }
        else
        {
          // Recurs down
          //debugMessage("↓ Recurs down\n");
          return insertRec ( cont, node->children[child], minDist );
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
        //debugMessage("+ Leaf so add here\n");
        node->cont      = cont;
        cont->qParent   = node;
        return VAL_ADDED;
      }
      else
      {
        //debugMessage("Not leaf\n");
        uint child = getPos(cont, node);

        if ( node->children[child] == NULL ) // Create the child node  .
        {
          //debugMessage("↓ Create new child for current candidate\n");

          createChild(child, node, cont);
          return VAL_ADDED;
        }
        else
        {
          // Recurs down
          //debugMessage("↓ Recurs down\n");
          return insertRec ( cont, node->children[child], minDist );
        }
      }
    }
  }
  else
  {
    if ( node->parent )
    {
      // Recurs UP
      //debugMessage("↑ Recurs UP\n");
      return insertRec ( cont, node->parent, minDist );
    }
    else
    {
      //debugMessage("Create new qHead\n");

      node->parent = new candQuadNode;

      //      node->parent->parent    = NULL;
      //      node->parent->cont      = NULL;
      //      node->parent->parent    = NULL;
      //
      //      for ( int i = 0; i < NO_QUAD_CHILD;i++ )
      //      {
      //        node->parent->children[i] = NULL;
      //      }

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
      //debugMessage("↑ Recurs UP\n");
      return insertRec ( cont, node->parent, minDist );
    }
  }
}

candQuadNode* candTree::createChild(uint child, candQuadNode* node, container* cont )
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

  return node->children[child];
}

container* candTree::getNeighbour(double val, valNode* node)
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
        if ( cand->larger == NULL )
        {
          return cand;
        }
        cand = cand->larger;
      }
      return cand;
    }
    else if ( node->vMax < val )
    {
      container* cand = node->max;

      while ( val > *cand )
      {
        if ( cand->larger == NULL )
        {
          return cand;
        }
        cand = cand->larger;
      }
      return cand;
    }
    else if ( node->vMin > val )
    {
      container* cand = node->min;

      while ( val < *cand )
      {
        if ( cand->smaller == NULL )
        {
          return cand;
        }
        cand = cand->smaller;
      }
      return cand;
    }
  }
  return NULL; // Probably can't get here with the noEls
}

int candTree::insertRec ( container* cont, valNode* node, double minDist )
{
  if ( node->noEls == 0 && !( node->max == NULL && node->max == NULL) )
  {
    errMsg("We have a problem!\n");
  }

  if ( *cont < *node )
  {
    bool leaf = true;

    for ( int i = 0; i < NO_VAL_CHILD; i++ )
    {
      if ( node->children[i] )
      {
        leaf = false;
        break;
      }
    }

    if ( leaf )
    {
      if ( node->noEls < MAX_VAL_LIST )
      {
        if ( node->noEls <= 0 )
        {
          //debugMessage("No els in node search for neighbour.\n");
          container* close = getNeighbour(cont->val, vHead);

          if ( close )
          {
            if ( *close < *cont )
            {
              if (  close->larger )
              {
                if ( !( *cont < *close->larger) )
                {
                  errMsg("ERROR: value not between bounds\n");
                }
              }
              else
              {
                //debugMessage("← END\n");
              }

              container* hold = close->larger;
              close->larger = cont;
              cont->smaller = close;
              if ( hold )
                hold->smaller = cont;
              cont->larger = hold;
            }
            else
            {
              if ( close->smaller )
              {
                if ( !( *close->smaller < *cont) )
                {
                  errMsg("ERROR: value not between bounds\n");
                }
              }
              else
              {
                //debugMessage("→ BEGIN\n");
              }

              container* hold = close->smaller;
              close->smaller = cont;
              cont->larger = close;
              if ( hold )
                hold->larger = cont;
              cont->smaller = hold;
            }
          }
        }

        uint res = insertIntoList(node->min, node->max, cont);

        cont->vParent = node;

        if ( res & NEW_HEAD )
        {

          node->min = cont;
        }

        if ( res & NEW_TAIL )
        {

          node->max = cont;
        }

        node->noEls++;

        return 1;
      }
      else  // This node has too many elements split it and add cont to relevant child  .
      {
        if ( (node->min->val == node->max->val) && (node->min->val == cont->val) )
        {
          uint res = insertIntoList(node->min, node->max, cont);

          cont->vParent = node;

          if ( res & NEW_HEAD )
          {

            node->min = cont;
          }

          if ( res & NEW_TAIL )
          {

            node->max = cont;
          }

          node->noEls++;

          return 1;
        }
        else
        {
          split( node );

          for ( int i = 0; i < NO_VAL_CHILD;i++ )
          {
            if ( *cont < *node->children[i] )
            {
              // Recurse down
              //debugMessage("↓ Recurs down  %7.3f  into [%.3f- %.3f] \n", cont->val, node->children[i]->vMin, node->children[i]->vMax);
              uint res = insertRec( cont, node->children[i], minDist );

              if ( res == 1 )
                node->noEls++;

              return res;
            }
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

        if ( child & TREE_RIGHT )
        {
          down->vMin = mid;
          down->vMax =  node->vMax;
        }
        else
        {
          down->vMin = node->vMin;
          down->vMax = mid;
        }

        down->parent = node;

        node->children[child] = down;

        //debugMessage("  Create new child [%7.3f - %7.3f].\n", down->vMin, down->vMax );
      }

      // Recurse down
      //debugMessage("↓ Recurs down  %7.3f  into [%.3f- %.3f] \n", cont->val, node->children[child]->vMin, node->children[child]->vMax);
      uint res = insertRec ( cont, node->children[child], minDist );

      if ( res == 1 )
        node->noEls++;

      return res;
    }
  }
  else // Not in this node recurse up  larger
  {
    valNode* up   = new valNode;
    up->noEls     = node->noEls;
    node->parent  = up;
    vHead         = up;

    if ( cont->val < node->vMin )
    {
      up->vMin = node->vMin - ( node->vMax - node->vMin );
      up->vMax = node->vMax;
      up->children[1] = node;
    }
    else
    {
      up->vMin = node->vMin;
      up->vMax = node->vMax + ( node->vMax - node->vMin );
      up->children[0] = node;
    }

    // Recurse up
    //debugMessage("↑ Recurs UP\n");
    uint res = insertRec ( cont, vHead, minDist );

    return res;
  }

  return 0;
}

container* candTree::insertDecreasing(container* head, container* cont)
{
  if ( head == NULL )
    return cont;

  container* mark = head;

  while ( *mark > *cont )
  {
    if ( mark->smaller == NULL )
    {
      container* hold =  mark->smaller;
      mark->smaller  = cont;
      cont->smaller  = hold;
      cont->larger    = mark;
      if( hold )
        hold->larger  = cont;

      return head;
    }
    mark = mark->smaller;
  }

  container* hold =  mark->larger;
  mark->larger        = cont;
  cont->larger        = hold;
  cont->smaller      = mark;
  if( hold )
    hold->smaller    = cont;

  if ( mark == head )
    return cont;
  else
    return head;

}

uint candTree::insertIntoList(container* head, container* tail, container* cont)
{
  //debugMessage("insert value into list\n");
  uint res = 0;

  if ( head == NULL && tail == NULL )
  {
    res |= NEW_HEAD;
    res |= NEW_TAIL;

    //cont->smaller  = NULL;
    //cont->larger    = NULL;

    return res;
  }

  container* mark = head;

  while ( *mark < *cont )
  {
    if ( mark == tail )
    {
      res |= NEW_TAIL; // This should always be the case
    }

    if ( mark->larger == NULL )
    {
      container* hold     =  mark->larger;
      mark->larger        = cont;
      cont->larger        = hold;
      cont->smaller       = mark;
      if( hold )
        hold->smaller     = cont;

      return res;
    }
    mark = mark->larger;
  }

  if ( mark == head )
  {
    res |= NEW_HEAD;
  }

  container* hold   =  mark->smaller;
  mark->smaller     = cont;
  cont->smaller     = hold;
  cont->larger      = mark;
  if( hold )
    hold->larger    = cont;

  return res;
}

uint candTree::candTree::split( valNode* node )
{
  //debugMessage("Split  %8.4f - %8.4f  \n", node->vMin, node->vMax );

  valNode* left   = new valNode;
  valNode* right  = new valNode;

  node->children[0] = left;
  node->children[1] = right;

  double mid = ( node->vMin + node->vMax ) / 2.0 ;

  left->vMin    = node->vMin;
  left->vMax    = mid;
  right->vMin   = mid;
  right->vMax   = node->vMax;

  container* cont = node->min;

  FOLD //  loop through left elements  .
  {
    while ( *cont < *left )
    {
      left->noEls++;
      cont->vParent = left;

      cont = cont->larger;
      if ( cont == NULL )
      {
        if ( left->noEls != node->noEls )
        {
          errMsg( "ERRR: wrong number of elements in node!\n");
        }
        break;
      }
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
    left->max       = cont->smaller;

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

  FOLD // Set right parent  .
  {
    uint cnt = 0;

    cont = right->min;

    while ( cont != right->max )
    {
      cont->vParent = right;
      cont = cont->larger;
      cnt++;
    }
    if (cont)
    {
      cont->vParent = right;
      cnt++;
    }
  }

  left->parent  = node;
  right->parent = node;

  node->min = NULL;
  node->max = NULL;

  return 0;
}

container* candTree::getRec( container* cont, candQuadNode* node, double dist )
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
              errMsg("ERROR node with cand has child?");
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
  return NULL;
}

container* candTree::getAllRec( container* cont, candQuadNode* node, boundingBox *bb, double dist, container* list)
{
  if (node)
  {
    if ( *bb & *node )
    {
      //debugMessage(" x [ %11.3f %11.3f ] Y [ %11.3f %11.3f ]\n",node->xMin, node->xMax, node->yMax, node->yMax );

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

container* candTree::getClosestRec( container* cont, candQuadNode* node, boundingBox *bb, double dist, container* closest)
{
  if (node)
  {
    if ( *bb & *node )
    {
      //double tmpDST = *cont | *node ;
      //tmpDST = sqrt(tmpDST);
      //debugMessage(" x [ %11.3f %11.3f ] Y [ %11.3f %11.3f ]   %16.3f\n",node->xMin, node->xMax, node->yMin, node->yMax, tmpDST );

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
              // Within distance replace closes
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

container* candTree::getLargestRec( container* cont, candQuadNode* node, boundingBox *bb, double dist, container* largest)
{
  if (node)
  {
    if ( *bb & *node )
    {
      for ( int i = 0; i < NO_QUAD_CHILD; i++ )
      {
        if ( node->children[i] )
        {
          if ( *bb & *node->children[i] )
          {
            largest = getLargestRec( cont, node->children[i], bb, dist, largest);
          }
        }
      }

      // this is a leaf so search my value!
      if ( node->cont )
      {
        double newDst = *cont | *node->cont ;

        if ( newDst <= dist )
        {
          if ( !largest )
          {
            return node->cont;
          }
          else if ( *node->cont > *largest )
          {
            // Larger so replace
            largest = node->cont;
          }
        }
      }
    }
  }
  return largest;
}

uint candTree::countRec( candQuadNode* node)
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
        errMsg("ERROR: Node with cand that is not a leaf!\n");
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

uint candTree::getPos( container* cont, candQuadNode* node )
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

uint candTree::checkVal()
{
  container* cont = getLargest(vHead);
  uint cnt = 0;

  while(cont)
  {
    if (cont->smaller)
    {
      if ( *cont < *cont->smaller )
      {
        errMsg("ERROR: values not in order!\n");
      }
    }
    cnt++;
    cont = cont->smaller;
  }

  if ( cnt != vHead->noEls )
  {
    errMsg("ERROR: Wrong number of elements!\n");
  }

  return 0;
}

uint candTree::check(valNode* node)
{
  if ( node->min || node->max )
  {
    if ( node->min == NULL || node->max == NULL )
      errMsg("ERROR: node->min == NULL || node->max == NULL \n");

    container* cont = node->min;
    uint cnt = 0;
    FOLD //  loop through left elements  .
    {
      while ( cont != node->max )
      {
        cnt++;
        if ( cont->vParent != node )
          errMsg("ERROR: cont->vParent != node \n");

        cont = cont->larger;
      }
      if ( cont == node->max )
      {
        cnt++;
        if ( cont->vParent != node )
          errMsg("ERROR: cont->vParent != node \n");

      }

      if ( cnt != node->noEls )
        errMsg("ERROR: cnt != node->noEls \n");
    }
  }

  for ( int i = 0; i < NO_VAL_CHILD;i++ )
  {
    if ( node->children[i] )
    {
      check(node->children[i]);
    }
  }

  return 0;
}

uint candTree::check(candQuadNode* node)
{
  if ( node )
  {
    if ( node->cont )
    {
      if ( node->cont->qParent != node )
        errMsg("ERROR: node->cont->qParent != node \n");
    }
  }

  for ( int i = 0; i < NO_QUAD_CHILD;i++ )
  {
    if ( node->children[i] )
    {
      check(node->children[i]);
    }
  }

  return 0;
}

uint candTree::decValCount(valNode* node)
{
  if ( node )
  {
    node->noEls --;
    decValCount(node->parent);
  }

  return 0;
}

uint candTree::removeActula( container* cont )
{
  //debugMessage("Remove:  Value: %.4f  X: %.4f  Y: %.4f \n", cont->val, cont->x, cont->y );

  if ( cont->larger )
    cont->larger->smaller = cont->smaller;

  if ( cont->smaller )
    cont->smaller->larger = cont->larger;

  if      ( !(cont->larger) && !(cont->smaller) )
  {
    // This is the only container in a list
    if ( cont->vParent )
    {
      // Clear parent node pointers
      if ( cont->vParent->min == cont &&  cont->vParent->max == cont )
      {
        cont->vParent->min = NULL;
        cont->vParent->max = NULL;
      }
    }
  }
  else if ( cont->larger && cont->smaller )
  {
    // This container is in a value list
    if ( cont->vParent->min == cont->vParent->max )
    {
      // This node was the only one its parent had
      cont->vParent->min = NULL;
      cont->vParent->max = NULL;
    }
    else
    {
      // It is in a list
      if ( cont->larger->vParent != cont->smaller->vParent )
      {
        // This was one of the ends of a list
        if ( cont->vParent->min == cont )
        {
          if ( cont->larger->vParent == cont->vParent )
          {
            // My neighbour is now the end of the list
            cont->vParent->min = cont->larger;
          }
          else
          {
            // This must have been the only container in the list of this node
            cont->vParent->min = NULL;
            cont->vParent->max = NULL;
          }
        }

        if ( cont->vParent->max == cont )
        {
          if ( cont->smaller->vParent == cont->vParent )
          {
            // My neighbour is now the end of the list
            cont->vParent->max = cont->smaller;
          }
          else
          {
            // This must have been the only container in the list of this node
            cont->vParent->min = NULL;
            cont->vParent->max = NULL;
          }
        }
      }
    }
  }
  else
  {
    // This was the end of the global list
    if ( cont->larger )
    {
      if ( cont->vParent == cont->larger->vParent)
      {
        // My neighbour is now the end of the list
        cont->vParent->min = cont->larger ;
      }
      else
      {
        // This node was the only one its parent had
        cont->vParent->max = NULL;
        cont->vParent->min = NULL;
      }
    }

    if ( cont->smaller )
    {
      if ( cont->vParent == cont->smaller->vParent)
      {
        // My neighbour is now the end of the list
        cont->vParent->max = cont->smaller ;
      }
      else
      {
        // This node was the only one its parent had
        cont->vParent->max = NULL;
        cont->vParent->min = NULL;
      }
    }

  }

  FOLD //  Set descendants of parents  .
  {
    if ( cont->vParent )
    {
      decValCount(cont->vParent);

      if ( cont->vParent->noEls == 0 && !(cont->vParent->max == NULL && cont->vParent->max == NULL) ) // TMP  .
      {
        errMsg( "We have a problem!\n");
      }
    }

    if ( cont->qParent )
    {
      if ( cont->qParent->cont == cont )
      {
        cont->qParent->cont   = NULL;
        cont->qParent         = NULL; // Not strictly necessary as we are going to delete this anyway
      }
      else
      {
        errMsg( "WARNING: Quadtree node not expecting this container.\n");
      }
    }
  }

  freeNull(cont->data);
  freeNull(cont);

  return 0;
}

uint candTree::removeMarkedA()
{
  uint noremoved = 0;

  container* cont = getSmallest();
  container* next;
  while (cont)
  {
    next = cont->larger;
    if (cont->flag & REMOVE_CONTAINER )
    {
      removeActula(cont);
      noremoved++;
    }
    cont = next;
  }

  return noremoved;
}

void candTree::freeRec(valNode* node)
{
  for ( int i = 0; i < NO_VAL_CHILD;i++ )
  {
    if ( node->children[i] )
    {
      freeRec(node->children[i]);
    }
  }

  freeNull(node);
}

void candTree::freeRec(candQuadNode* node)
{
  for ( int i = 0; i < NO_QUAD_CHILD;i++ )
  {
    if ( node->children[i] )
    {
      freeRec(node->children[i]);
    }
  }

  freeNull(node);
}
