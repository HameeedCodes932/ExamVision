import type { ReactNode } from "react";

interface DashboardLayoutProps {
  header: ReactNode;
  cameraFeed: ReactNode;
  eventFeed: ReactNode;
  studentGrid: ReactNode;
  alertPanel: ReactNode;
  examSession: ReactNode;
}

export function DashboardLayout({
  header,
  cameraFeed,
  eventFeed,
  studentGrid,
  alertPanel,
  examSession,
}: DashboardLayoutProps) {
  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-3 flex flex-col gap-3">
      {header}

      <div className="grid grid-cols-1 xl:grid-cols-5 gap-3 flex-1 min-h-0">
        {/* Left column: camera + students */}
        <div className="xl:col-span-2 flex flex-col gap-3 min-h-0">
          {cameraFeed}
          <div className="flex-1 min-h-0">{studentGrid}</div>
        </div>

        {/* Right column: events + alerts + exam */}
        <div className="xl:col-span-3 flex flex-col gap-3 min-h-0">
          {examSession}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-3 flex-1 min-h-0">
            {eventFeed}
            {alertPanel}
          </div>
        </div>
      </div>
    </div>
  );
}
