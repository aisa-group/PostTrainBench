#!/bin/bash

NUM_HOURS=$1
TARGET_PATH=$2
CREATION_DATE=$(date +%s)

cat > "$TARGET_PATH" << TIMER
#!/bin/bash

NUM_HOURS=${NUM_HOURS}
CREATION_DATE=${CREATION_DATE}

DEADLINE=\$((CREATION_DATE + NUM_HOURS * 3600))
NOW=\$(date +%s)
REMAINING=\$((DEADLINE - NOW))

if [ \$REMAINING -le 0 ]; then
    echo "Timer expired!"
else
    echo "Remaining time (hours:minutes)":
    HOURS=\$((REMAINING / 3600))
    MINUTES=\$(((REMAINING % 3600) / 60))
    printf "%d:%02d\n" \$HOURS \$MINUTES
fi
TIMER

chmod +x "$TARGET_PATH"