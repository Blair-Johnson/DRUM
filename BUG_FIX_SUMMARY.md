# Bug Fix Summary: Database Indexing and Validation Issues

## Issues Reported

1. **Validation metrics showing despite no validation split**: User saw validation loss and batch numbers even though dataset had no validation split
2. **Rules using predominantly single predicate type**: Learned rules almost exclusively used patterns like `pred1(x,z), pred1(z,y)` with no diversity of predicates

## Root Causes Identified

### Issue 1: Validation Auto-Split
- Small Prolog datasets were being auto-split into train/validation even with only 6 examples
- Random chance determined whether validation got examples (seed-dependent)
- Confusing UX when user didn't explicitly provide validation data

### Issue 2: Critical Database Indexing Bug

**The Problem:**
When `restrict_domain=True`, there was a critical mismatch between model expectations and database structure:

- **Model side**: Expected operators 0-3 to map to 4 background relations
  - Operator 0 → brother
  - Operator 1 → father  
  - Operator 2 → mother
  - Operator 3 → sister

- **Data side**: Database indexed by FULL vocabulary (0-7 for all 8 relations)
  - Database[0] → aunt
  - Database[1] → brother
  - Database[2] → father
  - Database[3] → grandfather
  - ...

**The Impact:**
- When model tried to use "brother" (operator 0), it actually accessed "aunt" facts (database[0])
- When model tried to use "father" (operator 1), it actually accessed "brother" facts (database[1])
- Model was learning with completely WRONG relations!
- This caused poor rule quality and inability to learn diverse multi-predicate rules

## Solutions Implemented

### Fix 1: Improved Validation Handling
- **Change**: Only auto-split validation if dataset has ≥20 training examples
- **File**: `src/data.py` line 136-142
- **Result**: Small Prolog datasets no longer show spurious validation metrics

### Fix 2: Database Index Remapping
- **Change**: Created `operator_to_relation` mapping and modified `_db_to_matrix_db()`
- **Files**: `src/data.py` lines 167-177, 391-421
- **How it works**:
  1. Create mapping: operator_idx → full_vocab_idx for background relations
     - {0→1(brother), 1→2(father), 2→5(mother), 3→6(sister)}
  2. Database now indexed by operator indices (0-3), not full vocab (0-7)
  3. Each database[i] contains facts for operator i's corresponding relation
- **Result**: Model now accesses correct relations during training

## Verification

### Before Fix:
```
Database[0] = aunt (full vocab index 0)
Database[1] = brother (full vocab index 1)
Database[2] = father (full vocab index 2)
Database[3] = grandfather (full vocab index 3)
```
Model operator 0 trying to use "brother" → accesses aunt facts ❌

### After Fix:
```
Database[0] = brother (remapped from full vocab index 1)
Database[1] = father (remapped from full vocab index 2)
Database[2] = mother (remapped from full vocab index 5)
Database[3] = sister (remapped from full vocab index 6)
```
Model operator 0 trying to use "brother" → accesses brother facts ✓

## Expected Improvements

With the database indexing fix:
- ✅ Model can now access correct background relations
- ✅ Should learn proper multi-hop rules (e.g., `grandfather(X,Y) :- father(X,Z), father(Z,Y)`)
- ✅ Rules should show diversity of predicates within rule bodies
- ✅ No more mysterious single-predicate patterns

## Testing

Comprehensive tests verify:
1. ✅ Database keys match expected operator indices [0, 1, 2, 3]
2. ✅ Each database entry contains facts for correct relation
3. ✅ Operator-to-relation mapping works correctly
4. ✅ TSV format backward compatibility maintained
5. ✅ Small datasets don't auto-split validation
6. ✅ Training examples properly excluded from database when restrict_domain=True

## Files Modified

- `src/data.py`: 
  - Added `operator_to_relation` mapping creation
  - Modified `_db_to_matrix_db()` to remap database indices
  - Changed validation auto-split threshold to 20 examples

## Next Steps

User should:
1. Retrain model on test_prolog dataset
2. Extract rules and verify diverse predicate usage
3. Rules like `aunt(X,Y) :- sister(X,Z), father(Z,Y)` should now appear
4. If still seeing single-predicate rules, may need to adjust training hyperparameters (learning rate, epochs, etc.)
