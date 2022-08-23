export function detectCollision(vehicle, gameObject) {
  if (
    vehicle.goal.x === gameObject.position.x &&
    vehicle.goal.y === gameObject.position.y &&
    gameObject.lastCol.x != gameObject.position.x &&
    gameObject.lastCol.y != gameObject.position.y
  ) {
    gameObject.lastCol = { x: gameObject.position.x, y: gameObject.position.y };
    return true;
  }

  return false;
}
