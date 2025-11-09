#!/bin/bash
# Quick script to check database mission storage

echo "🔍 Checking Database Mission Storage..."
echo ""

# Check mission counts
echo "📊 Mission Counts:"
docker exec roverops-mysql mysql -u roverops_user -proverops_password roverops_db -e "
SELECT 
    'missions' as table_name, COUNT(*) as count FROM missions
UNION ALL
SELECT 'mission_steps', COUNT(*) FROM mission_steps
UNION ALL
SELECT 'mission_logs', COUNT(*) FROM mission_logs
UNION ALL
SELECT 'mission_obstacles', COUNT(*) FROM mission_obstacles
UNION ALL
SELECT 'mission_reports', COUNT(*) FROM mission_reports;
" 2>/dev/null

echo ""
echo "📋 Recent Missions:"
docker exec roverops-mysql mysql -u roverops_user -proverops_password roverops_db -e "
SELECT 
    mission_id,
    LEFT(goal, 40) as goal,
    status,
    rover_final_x,
    rover_final_y,
    created_at
FROM missions
ORDER BY created_at DESC
LIMIT 5;
" 2>/dev/null

echo ""
echo "📄 Mission Reports:"
docker exec roverops-mysql mysql -u roverops_user -proverops_password roverops_db -e "
SELECT 
    mission_id,
    mission_status,
    completed_steps,
    total_steps,
    created_at
FROM mission_reports
ORDER BY created_at DESC
LIMIT 5;
" 2>/dev/null

echo ""
echo "📝 Recent Activity Logs:"
docker exec roverops-mysql mysql -u roverops_user -proverops_password roverops_db -e "
SELECT 
    mission_id,
    agent_type,
    level,
    LEFT(message, 60) as message,
    timestamp
FROM mission_logs
ORDER BY timestamp DESC
LIMIT 10;
" 2>/dev/null

