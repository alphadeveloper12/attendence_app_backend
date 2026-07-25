import { BrowserRouter, Routes, Route } from 'react-router-dom'
import HomePage from './home/HomePage'
import LoginPage from './pages/LoginPage'
import DashboardLayout from './app/DashboardLayout'
import { PlaceholderPage } from './app/PlaceholderPage'
import AttritionRiskPage from './features/analytics/AttritionRiskPage'
import DocumentExpiryPage from './features/analytics/DocumentExpiryPage'
import ManpowerRecommendationsPage from './features/analytics/ManpowerRecommendationsPage'
import AskDataPage from './features/analytics/AskDataPage'
import DistributionPage from './features/distribution/DistributionPage'
import DepartmentsPage from './features/departments/DepartmentsPage'
import SitesPage from './features/sites/SitesPage'
import SiteDetailPage from './features/sites/SiteDetailPage'
import FaceEnrollmentPage from './features/face/FaceEnrollmentPage'
import GeofenceTuningPage from './features/geofence/GeofenceTuningPage'
import SiteAdminsPage from './features/siteAdmins/SiteAdminsPage'
import SettingsPage from './features/settings/SettingsPage'
import DashboardPage from './features/dashboard/DashboardPage'
import ReportsPage from './features/reports/ReportsPage'
import MonthlyReportPage from './features/monthly/MonthlyReportPage'
import SalaryReportPage from './features/salary/SalaryReportPage'
import UserDetailPage from './features/userDetail/UserDetailPage'
import EmployeeEditPage from './features/dashboard/EmployeeEditPage'

/**
 * Top-level routing.
 *  - `/`, `/home-v2/`  → marketing home
 *  - `/login-v2/`      → login
 *  - dashboard pages   → each mirrors its legacy path with a `-v2` suffix
 *    (e.g. /dashboard-v2, /dashboard/reports-v2). They render inside the
 *    pathless `DashboardLayout` route. Django serves the SPA shell for any
 *    `-v2` URL; React Router picks the page.
 *
 * Pages are filled in phase by phase (placeholders for now).
 */
export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/home-v2/" element={<HomePage />} />
        <Route path="/login-v2/" element={<LoginPage />} />

        {/* Dashboard — pathless layout route so children can use absolute -v2 paths */}
        <Route element={<DashboardLayout />}>
          <Route path="/dashboard-v2" element={<DashboardPage />} />
          <Route path="/dashboard/reports-v2" element={<ReportsPage />} />
          <Route path="/dashboard/monthly-report-v2" element={<MonthlyReportPage />} />
          <Route path="/dashboard/distribution-list-v2" element={<DistributionPage />} />
          <Route path="/dashboard/departments-management-v2" element={<DepartmentsPage />} />
          <Route path="/dashboard/user-face-v2" element={<FaceEnrollmentPage />} />
          <Route path="/dashboard/attrition-risk-v2" element={<AttritionRiskPage />} />
          <Route path="/dashboard/document-expiry-v2" element={<DocumentExpiryPage />} />
          <Route path="/dashboard/manpower-recommendations-v2" element={<ManpowerRecommendationsPage />} />
          <Route path="/dashboard/ask-data-v2" element={<AskDataPage />} />
          <Route path="/dashboard/geofence-tuning-v2" element={<GeofenceTuningPage />} />
          <Route path="/dashboard/salary-report-v2" element={<SalaryReportPage />} />
          <Route path="/dashboard/sites-v2" element={<SitesPage />} />
          <Route path="/dashboard/sites/:idv2" element={<SiteDetailPage />} />
          <Route path="/dashboard/site-admins-v2" element={<SiteAdminsPage />} />
          <Route path="/dashboard/settings-v2" element={<SettingsPage />} />
          <Route path="/dashboard/employee-edit/:idv2" element={<EmployeeEditPage />} />
          <Route path="/dashboard/user/:idv2" element={<UserDetailPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
