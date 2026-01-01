import { AppShell } from '@/components/shell/AppShell';

export default function AutoPilotLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <AppShell>{children}</AppShell>;
}
