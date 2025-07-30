// src/components/ManualScrollDiv.tsx
import React, { useRef, useEffect } from "react";

// ManualScrollDiv Component with momentum scrolling and boundary pass-through
const ManualScrollDiv: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({
  children,
  ...props
}) => {
  const scrollRef = useRef<HTMLDivElement>(null);
  const startY = useRef<number>(0);
  const startScrollTop = useRef<number>(0);
  const lastY = useRef<number>(0);
  const lastTime = useRef<number>(0);
  const velocity = useRef<number>(0);
  const momentumFrame = useRef<number | null>(null);

  const handleTouchStart = (e: TouchEvent) => {
    if (momentumFrame.current) {
      cancelAnimationFrame(momentumFrame.current);
      momentumFrame.current = null;
    }
    startY.current = e.touches[0].clientY;
    lastY.current = e.touches[0].clientY;
    startScrollTop.current = scrollRef.current
      ? scrollRef.current.scrollTop
      : 0;
    lastTime.current = e.timeStamp;
    velocity.current = 0;
    e.stopPropagation();
  };

  const handleTouchMove = (e: TouchEvent) => {
    const currentY = e.touches[0].clientY;
    const currentTime = e.timeStamp;
    const deltaY = currentY - startY.current;

    if (scrollRef.current) {
      const newScrollTop = startScrollTop.current - deltaY;
      const maxScrollTop =
        scrollRef.current.scrollHeight - scrollRef.current.clientHeight;
      if (newScrollTop > 0 && newScrollTop < maxScrollTop) {
        scrollRef.current.scrollTop = newScrollTop;
        e.preventDefault();
        e.stopPropagation();
      } else {
        if (newScrollTop < 0) {
          scrollRef.current.scrollTop = 0;
        } else if (newScrollTop > maxScrollTop) {
          scrollRef.current.scrollTop = maxScrollTop;
        }
      }
    }
    const dt = currentTime - lastTime.current;
    if (dt > 0) {
      velocity.current = (lastY.current - currentY) / dt;
    }
    lastY.current = currentY;
    lastTime.current = currentTime;
  };

  const momentum = () => {
    if (!scrollRef.current) return;
    const friction = 0.95;
    if (Math.abs(velocity.current) < 0.02) return;
    scrollRef.current.scrollTop += velocity.current * 16;
    velocity.current *= friction;
    if (
      scrollRef.current.scrollTop <= 0 ||
      scrollRef.current.scrollTop >=
        scrollRef.current.scrollHeight - scrollRef.current.clientHeight
    ) {
      return;
    }
    momentumFrame.current = requestAnimationFrame(momentum);
  };

  const handleTouchEnd = (_e: TouchEvent) => {
    momentumFrame.current = requestAnimationFrame(momentum);
  };

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    // Check if it's a touch device before adding listeners
    const isTouchDevice =
      "ontouchstart" in window || navigator.maxTouchPoints > 0;
    if (isTouchDevice) {
      el.addEventListener("touchstart", handleTouchStart, { passive: false });
      el.addEventListener("touchmove", handleTouchMove, { passive: false });
      el.addEventListener("touchend", handleTouchEnd, { passive: false });
    }
    return () => {
      if (isTouchDevice && el) {
        el.removeEventListener("touchstart", handleTouchStart);
        el.removeEventListener("touchmove", handleTouchMove);
        el.removeEventListener("touchend", handleTouchEnd);
        if (momentumFrame.current) {
          cancelAnimationFrame(momentumFrame.current);
        }
      }
    };
  }, []);

  return (
    <div ref={scrollRef} {...props}>
      {children}
    </div>
  );
};

export default ManualScrollDiv;
